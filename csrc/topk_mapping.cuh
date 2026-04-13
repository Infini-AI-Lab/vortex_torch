#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================
// TopK bucket-sort Stage-1 remap transforms (lean version).
//
// These are element-wise transforms applied to scores before
// the Stage-1 8-bit histogram bucketing. The goal is to spread
// a skewed raw distribution more uniformly across the 256 bins
// so the threshold bin shrinks and Stage-2 refinement does less
// work. Stage 2 still uses convert_to_uint32() on the remapped
// value's raw bits for tie-breaking.
//
// There is no pre-pass, no auto-range, no LUT, no quantile
// table, and no shared-memory state — each transform is a
// pure function of one float. The heavy pre-pass machinery
// (auto-range, pivot, tail-window, topk-window, LUT_CDF,
// QUANTILE, SUBTRACT, TRUNC8) lives in
// csrc/archived/fast_topk_vortex_prepass.cu.
// ============================================================

enum TopKMappingMode {
    MAPPING_NONE     = 0,  // identity (no remap)
    MAPPING_LUT_CDF  = 1,  // bin lookup: new_bin = lut[convert_to_uint8(x)]
    MAPPING_QUANTILE = 2,  // binary search over 256 calibrated quantile thresholds
    MAPPING_POWER    = 3,  // sign(x) * |x|^p
    MAPPING_LOG      = 4,  // sign(x) * log(|x| + 1)
    MAPPING_ASINH    = 6,  // asinh(beta * x)
    MAPPING_LOG1P    = 7,  // sign(x) * log1p(alpha * |x|)
    MAPPING_TRUNC8   = 8,  // identity bucketing (historical name, alias of MAPPING_NONE)
    MAPPING_ERF      = 9,  // erf(alpha * x)
    MAPPING_TANH     = 10, // tanh(alpha * x)
    MAPPING_SUBTRACT = 11, // x - pivot, with pivot = power_exp (free hyperparameter)
    MAPPING_EXP_STRETCH = 13, // exp(alpha * x)
};

struct TopKMappingParams {
    int   mode;       // TopKMappingMode
    float power_exp;  // Free hyperparameter: p / alpha / beta / pivot depending on mode
    const uint8_t* __restrict__ lut;       // [256] uint8 LUT, MAPPING_LUT_CDF only
    const float*   __restrict__ quantiles; // [256] float quantile breakpoints, MAPPING_QUANTILE only
};

// ---- Element-wise transforms ----

__device__ __forceinline__ float transform_power(float x, float p) {
    return copysignf(__powf(fabsf(x), p), x);
}

__device__ __forceinline__ float transform_log(float x) {
    return copysignf(__logf(fabsf(x) + 1.0f), x);
}

__device__ __forceinline__ float transform_asinh(float x, float beta) {
    return asinhf(beta * x);
}

__device__ __forceinline__ float transform_log1p(float x, float alpha) {
    return copysignf(log1pf(alpha * fabsf(x)), x);
}

__device__ __forceinline__ float transform_erf(float x, float alpha) {
    return erff(alpha * x);
}

__device__ __forceinline__ float transform_tanh(float x, float alpha) {
    return tanhf(alpha * x);
}

__device__ __forceinline__ float transform_exp_stretch(float x, float alpha) {
    float z = alpha * x;
    z = fminf(z, 80.0f);  // prevent float32 overflow (exp(80) ~ 5.5e34)
    return expf(z);
}

// Pure element-wise dispatcher. Returns the *float value* after the transform.
// For bin-selection modes (LUT_CDF / QUANTILE) this is identity: the mapping
// happens in compute_stage1_bin() below instead of via a float transform, so
// Stage-2 tie-breaking uses the raw score bits for those modes.
__device__ __forceinline__ float apply_transform(float x, const TopKMappingParams& params) {
    switch (params.mode) {
        case MAPPING_POWER:       return transform_power(x, params.power_exp);
        case MAPPING_LOG:         return transform_log(x);
        case MAPPING_ASINH:       return transform_asinh(x, params.power_exp);
        case MAPPING_LOG1P:       return transform_log1p(x, params.power_exp);
        case MAPPING_ERF:         return transform_erf(x, params.power_exp);
        case MAPPING_TANH:        return transform_tanh(x, params.power_exp);
        case MAPPING_SUBTRACT:    return x - params.power_exp;
        case MAPPING_EXP_STRETCH: return transform_exp_stretch(x, params.power_exp);
        case MAPPING_LUT_CDF:
        case MAPPING_QUANTILE:
        case MAPPING_TRUNC8:
        default:                  return x;  // NONE / TRUNC8 / LUT_CDF / QUANTILE
    }
}

// Whether the mapping mode is a direct bin-selection function (LUT_CDF /
// QUANTILE). These modes need per-block shared-memory tables.
__device__ __forceinline__ bool mapping_uses_table(int mode) {
    return mode == MAPPING_LUT_CDF || mode == MAPPING_QUANTILE;
}

// Binary search over a sorted [256] quantile table. Returns the largest
// index i such that x >= quantiles[i], in [0, 255].
__device__ __forceinline__ uint8_t quantile_bin_lookup(
    float x, const float* __restrict__ s_quantiles)
{
    int lo = 0, hi = 255;
#pragma unroll 8
    for (int iter = 0; iter < 8; ++iter) {
        int mid = (lo + hi + 1) >> 1;
        if (x >= s_quantiles[mid]) lo = mid;
        else hi = mid - 1;
    }
    return static_cast<uint8_t>(lo);
}

// Forward decl so compute_stage1_bin can call it. Defined in the enclosing TU.
__device__ __forceinline__ uint8_t convert_to_uint8(float x);

// Compute the Stage-1 bin for a raw score under any mapping mode. LUT_CDF /
// QUANTILE use the shared-memory tables loaded at the kernel entry; every
// other mode falls back to convert_to_uint8(apply_transform(x)).
__device__ __forceinline__ uint8_t compute_stage1_bin(
    float raw,
    const TopKMappingParams& params,
    const uint8_t* __restrict__ s_lut,
    const float*   __restrict__ s_quantiles)
{
    switch (params.mode) {
        case MAPPING_LUT_CDF:
            return s_lut[convert_to_uint8(raw)];
        case MAPPING_QUANTILE:
            return quantile_bin_lookup(raw, s_quantiles);
        default:
            return convert_to_uint8(apply_transform(raw, params));
    }
}
