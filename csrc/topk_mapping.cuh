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
    // MAPPING_LUT_CDF  = 1,  // bin lookup: new_bin = lut[convert_to_uint8(x)]
    // MAPPING_QUANTILE = 2,  // binary search over 256 calibrated quantile thresholds
    MAPPING_POWER    = 3,  // sign(x) * |x|^p
    MAPPING_LOG      = 4,  // sign(x) * log(|x| + 1)
    MAPPING_ASINH    = 6,  // asinh(beta * x)
    MAPPING_LOG1P    = 7,  // sign(x) * log1p(alpha * |x|)
    MAPPING_TRUNC8   = 8,  // identity bucketing (historical name, alias of MAPPING_NONE)
    MAPPING_ERF      = 9,  // erf(alpha * x)
    MAPPING_TANH     = 10, // tanh(alpha * x)
    MAPPING_SUBTRACT = 11, // x - pivot, with pivot = power_exp (free hyperparameter)
    MAPPING_EXP_STRETCH  = 13, // exp(alpha * x)
    // Top-spreading transforms (see CLAUDE.md / remap bench plan):
    // amplify differences in the high-score region so the top-K values
    // occupy multiple Stage-1 bins instead of collapsing into one.
    MAPPING_SHIFT_POW2   = 15, // sign(x - p) * (x - p)^2        [p = power_exp]
    MAPPING_SHIFT_POW3   = 16, // (x - p)^3                       [p = power_exp]
    MAPPING_LINEAR_STEEP = 17, // x + k * max(x, 0)               [k = power_exp]
    // One-sided spread: collapse below-pivot values into a single bin so
    // every above-pivot page gets its own slice of the 256-bin histogram.
    MAPPING_HALF_SQUARE  = 18, // max(x - p, 0)^2                 [p = power_exp]
    MAPPING_HALF_CUBE    = 19, // max(x - p, 0)^3                 [p = power_exp]
    // Bit-level remap: identity value transform, but the Stage-1 bucket
    // function in fast_topk_clean_fused switches to a mantissa-heavy bit
    // slice (bits [23:16] of convert_to_uint32) that gives 128 sub-bins
    // per exponent slot instead of 4. Zero per-element compute overhead;
    // the "remap" is the bucket change. Monotonic within 2 adjacent
    // fp32 exponent slots.
    // MAPPING_DENSE_MANT   = 20, // identity; bucketing handled in fused kernel
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

// Signed squared distance from a pivot. ~3 ops (1 sub, 1 mul, 1 copysign).
// Quadratically amplifies differences between values far from pivot so the
// top-K region gets spread across multiple Stage-1 bins.
__device__ __forceinline__ float transform_shift_pow2(float x, float pivot) {
    const float d = x - pivot;
    return copysignf(d * d, d);
}

// Signed cubic of distance from pivot. ~3 ops (1 sub, 2 mul; odd function so
// no copysign). Steeper growth than pow2 for even tighter top-K clusters.
__device__ __forceinline__ float transform_shift_pow3(float x, float pivot) {
    const float d = x - pivot;
    return d * d * d;
}

// Half-range linear stretch: positive values get multiplied by (1 + k),
// negative values pass through untouched. ~2 ops (fmax + fma). For softmax-
// style attention scores (which are non-negative after softmax), k = 8..16
// shifts the positive fp16 exponent up by 3..4 slots and empties out the
// collision at the top of the distribution.
__device__ __forceinline__ float transform_linear_steep(float x, float k) {
    return fmaf(k, fmaxf(x, 0.0f), x);
}

// One-sided shifted square: values below pivot collapse to 0 (they all end
// up in the same low Stage-1 bin), above-pivot values are squared so their
// differences amplify quadratically. ~2 ops (fmax + mul). The whole 256-bin
// histogram becomes dedicated to the top slice of the distribution.
__device__ __forceinline__ float transform_half_square(float x, float pivot) {
    const float d = fmaxf(x - pivot, 0.0f);
    return d * d;
}

// One-sided shifted cube: like half_square but cubic. ~3 ops. Best when the
// top-K region is even more tightly clustered and needs steeper amplification.
__device__ __forceinline__ float transform_half_cube(float x, float pivot) {
    const float d = fmaxf(x - pivot, 0.0f);
    return d * d * d;
}

// Compile-time templated dispatcher. When the caller knows the mapping mode
// at template-instantiation time, this lets the compiler fully inline the
// transform into the Stage-1 inner loop and eliminate the runtime switch
// that `apply_transform` would otherwise perform per element. Used by the
// per-mode specializations of `fast_topk_clean_fused` in topk_sglang.cu.
template <int MODE>
__device__ __forceinline__ float apply_transform_tmpl(float x, float p) {
    if      constexpr (MODE == MAPPING_POWER)        return transform_power(x, p);
    else if constexpr (MODE == MAPPING_LOG)          return transform_log(x);
    else if constexpr (MODE == MAPPING_ASINH)        return transform_asinh(x, p);
    else if constexpr (MODE == MAPPING_LOG1P)        return transform_log1p(x, p);
    else if constexpr (MODE == MAPPING_ERF)          return transform_erf(x, p);
    else if constexpr (MODE == MAPPING_TANH)         return transform_tanh(x, p);
    else if constexpr (MODE == MAPPING_SUBTRACT)     return x - p;
    else if constexpr (MODE == MAPPING_EXP_STRETCH)  return transform_exp_stretch(x, p);
    else if constexpr (MODE == MAPPING_SHIFT_POW2)   return transform_shift_pow2(x, p);
    else if constexpr (MODE == MAPPING_SHIFT_POW3)   return transform_shift_pow3(x, p);
    else if constexpr (MODE == MAPPING_LINEAR_STEEP) return transform_linear_steep(x, p);
    else if constexpr (MODE == MAPPING_HALF_SQUARE)  return transform_half_square(x, p);
    else if constexpr (MODE == MAPPING_HALF_CUBE)    return transform_half_cube(x, p);
    else                                             return x;  // NONE / TRUNC8
}

// Pure element-wise dispatcher. Returns the *float value* after the transform.
__device__ __forceinline__ float apply_transform(float x, const TopKMappingParams& params) {
    switch (params.mode) {
        case MAPPING_POWER:        return transform_power(x, params.power_exp);
        case MAPPING_LOG:          return transform_log(x);
        case MAPPING_ASINH:        return transform_asinh(x, params.power_exp);
        case MAPPING_LOG1P:        return transform_log1p(x, params.power_exp);
        case MAPPING_ERF:          return transform_erf(x, params.power_exp);
        case MAPPING_TANH:         return transform_tanh(x, params.power_exp);
        case MAPPING_SUBTRACT:     return x - params.power_exp;
        case MAPPING_EXP_STRETCH:  return transform_exp_stretch(x, params.power_exp);
        case MAPPING_SHIFT_POW2:   return transform_shift_pow2(x, params.power_exp);
        case MAPPING_SHIFT_POW3:   return transform_shift_pow3(x, params.power_exp);
        case MAPPING_LINEAR_STEEP: return transform_linear_steep(x, params.power_exp);
        case MAPPING_HALF_SQUARE:  return transform_half_square(x, params.power_exp);
        case MAPPING_HALF_CUBE:    return transform_half_cube(x, params.power_exp);
        case MAPPING_TRUNC8:
        default:                   return x;  // NONE / TRUNC8
    }
}

// Bin-selection table modes (LUT_CDF / QUANTILE) have been retired.
// This helper is kept for ABI compat with callers that still invoke it.
__device__ __forceinline__ bool mapping_uses_table(int /*mode*/) {
    return false;
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

// Compute the Stage-1 bin for a raw score. LUT_CDF / QUANTILE modes
// have been removed; every mode now goes through the element-wise
// apply_transform + convert_to_uint8.
__device__ __forceinline__ uint8_t compute_stage1_bin(
    float raw,
    const TopKMappingParams& params,
    const uint8_t* __restrict__ /*s_lut*/,
    const float*   __restrict__ /*s_quantiles*/)
{
    return convert_to_uint8(apply_transform(raw, params));
}
