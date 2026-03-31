#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================
// TopK bucket-sort distribution mapping strategies
//
// These transforms remap float scores before Stage 1's 8-bit
// histogram binning, aiming for a more uniform distribution
// across the 256 coarse bins.  Stage 2 refinement still uses
// convert_to_uint32() on raw floats, so correctness is preserved.
//
// Modes 3/4/6/7 use a data-adaptive linear mapping to [0,255]
// instead of fp16 bit-pattern bucketing, guaranteeing full
// bucket utilization regardless of value range.
// ============================================================

enum TopKMappingMode {
    MAPPING_NONE     = 0,  // Original convert_to_uint8 behavior
    MAPPING_LUT_CDF  = 1,  // LUT-based CDF equalization
    MAPPING_QUANTILE = 2,  // Piecewise-linear quantile mapping
    MAPPING_POWER    = 3,  // Monotonic power transform
    MAPPING_LOG      = 4,  // Log transform
    MAPPING_INDEX_CACHE = 5,  // Sentinel: reuse previous layer's indices (Python-level skip)
    MAPPING_ASINH       = 6,  // asinh(beta * x), beta via power_exp
    MAPPING_LOG1P       = 7,  // sign(x) * log1p(alpha * |x|), alpha via power_exp
    MAPPING_TRUNC8      = 8,  // BF16 upper-8-bit bucketing
};

struct TopKMappingParams {
    int mode;                              // TopKMappingMode
    float power_exp;                       // For MAPPING_POWER (default 0.5)
    const uint8_t* __restrict__ lut;       // [256] byte LUT, or nullptr
    const float* __restrict__ quantiles;   // [256] float quantile breakpoints, or nullptr
};

// NOTE: convert_to_uint8() must be defined before including this header.
// It is defined in topk_sglang.cu within the anonymous namespace.

// ---- Individual transform functions (return float, no bucketing) ----

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

// ---- Transform dispatcher (returns float, no bucketing) ----

__device__ __forceinline__ float apply_transform(float x, const TopKMappingParams& params) {
    switch (params.mode) {
        case MAPPING_POWER: return transform_power(x, params.power_exp);
        case MAPPING_LOG:   return transform_log(x);
        case MAPPING_ASINH: return transform_asinh(x, params.power_exp);
        case MAPPING_LOG1P: return transform_log1p(x, params.power_exp);
        default: return x;
    }
}

// ---- Linear bucketing for transform modes ----

__device__ __forceinline__ uint8_t linear_map_to_uint8(float val, float range_min, float inv_range) {
    int bin = __float2int_rd((val - range_min) * inv_range);
    return static_cast<uint8_t>(min(max(bin, 0), 255));
}

// ---- BF16 upper-8-bit bucketing (mode 8) ----

__device__ __forceinline__ uint8_t convert_to_uint8_bf16(float x) {
    __nv_bfloat16 bf = __float2bfloat16_rn(x);
    uint16_t bits = __bfloat16_as_ushort(bf);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                   : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

// ---- Non-transform mapping functions (unchanged) ----

// LUT-based CDF equalization: lut[original_bin] -> equalized_bin
__device__ __forceinline__ uint8_t map_lut_cdf(float x, const uint8_t* __restrict__ s_lut) {
    return s_lut[convert_to_uint8(x)];
}

// Quantile mapping: binary search over 256 sorted thresholds
__device__ __forceinline__ uint8_t map_quantile(float x, const float* __restrict__ s_quantiles) {
    // Binary search: find largest index i such that x >= s_quantiles[i]
    // s_quantiles is sorted ascending, length 256
    int lo = 0, hi = 255;
#pragma unroll 8
    for (int iter = 0; iter < 8; ++iter) {
        int mid = (lo + hi + 1) >> 1;
        if (x >= s_quantiles[mid]) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return static_cast<uint8_t>(lo);
}

// ---- Unified dispatcher ----
// For modes 3/4/6/7, range_min and inv_range come from a per-block pre-pass.

__device__ __forceinline__ uint8_t mapped_convert_to_uint8(
    float x,
    const TopKMappingParams& params,
    const uint8_t* __restrict__ s_lut,
    const float* __restrict__ s_quantiles,
    float range_min,
    float inv_range)
{
    switch (params.mode) {
        case MAPPING_LUT_CDF:
            if (params.lut != nullptr) return map_lut_cdf(x, s_lut);
            return convert_to_uint8(x);  // fallback to mode 0 when LUT not calibrated
        case MAPPING_QUANTILE:
            if (params.quantiles != nullptr) return map_quantile(x, s_quantiles);
            return convert_to_uint8(x);  // fallback to mode 0 when quantiles not calibrated
        case MAPPING_POWER:
        case MAPPING_LOG:
        case MAPPING_ASINH:
        case MAPPING_LOG1P: {
            float val = apply_transform(x, params);
            return linear_map_to_uint8(val, range_min, inv_range);
        }
        case MAPPING_TRUNC8:
            return convert_to_uint8_bf16(x);
        default:  // MAPPING_NONE
            return convert_to_uint8(x);
    }
}

// Helper: check if a mapping mode needs the auto-range pre-pass
__device__ __forceinline__ bool needs_auto_range(int mode) {
    return (mode == MAPPING_POWER || mode == MAPPING_LOG ||
            mode == MAPPING_ASINH || mode == MAPPING_LOG1P);
}
