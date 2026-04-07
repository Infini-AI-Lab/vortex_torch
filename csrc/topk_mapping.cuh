#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// ============================================================
// TopK bucket-sort Stage-1 remapping strategies
//
// These transforms remap float scores before Stage 1's 8-bit
// histogram binning.  The primary goal is to maximize coarse-bin
// resolution in the score region that determines the top-k
// cutoff, thereby:
//   - shrinking the Stage-1 threshold bin (fewer collisions)
//   - reducing COUNTER_NUM_EQUAL / COUNTER_STAGE2_INPUT
//   - reducing the number of Stage-2 refine rounds
//
// Stage 2 refinement still uses convert_to_uint32() on raw
// floats, so final ordering correctness is always preserved.
//
// Modes 3/4/6/7/9/10 apply a nonlinear transform then linearly
// map the result to [0,255].  Mode 12 (ADAPTIVE_TAIL_WINDOW)
// directly focuses all 256 bins on the competitive upper tail
// estimated from the top-k ratio, collapsing irrelevant
// low-score mass into bin 0.
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
    MAPPING_ERF         = 9,  // erf(alpha * x)
    MAPPING_TANH        = 10, // tanh(alpha * x)
    MAPPING_SUBTRACT    = 11, // subtract pivot, then fp16 bucketing
    MAPPING_ADAPTIVE_TAIL_WINDOW = 12, // focus bins on upper tail via sampled quantile
    MAPPING_EXP_STRETCH  = 13, // exp(alpha * x), concentrates bin resolution on upper tail
    MAPPING_TOPK_WINDOW  = 14, // k-aware linear windowing: focus bins on [tau_low, max]
};

struct TopKMappingParams {
    int mode;                              // TopKMappingMode
    float power_exp;                       // For MAPPING_POWER (default 0.5)
                                           // For MAPPING_ADAPTIVE_TAIL_WINDOW: tail expansion
                                           //   factor rho (default 4.0).  tau_low = Q(1 - rho*k/n).
    const uint8_t* __restrict__ lut;       // [256] byte LUT, or nullptr
    const float* __restrict__ quantiles;   // [256] float quantile breakpoints, or nullptr
    bool noscale;                          // Skip auto-range linear scaling, use fp16 bucketing on f(x)
    int sample_stride;                     // Pre-pass sampling stride (1=full, 8=1/8, 0=skip)
    int target_k;                          // Top-k value; used by MAPPING_ADAPTIVE_TAIL_WINDOW
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

// ---- Transform dispatcher (returns float, no bucketing) ----

__device__ __forceinline__ float apply_transform(float x, const TopKMappingParams& params) {
    switch (params.mode) {
        case MAPPING_POWER: return transform_power(x, params.power_exp);
        case MAPPING_LOG:   return transform_log(x);
        case MAPPING_ASINH: return transform_asinh(x, params.power_exp);
        case MAPPING_LOG1P: return transform_log1p(x, params.power_exp);
        case MAPPING_ERF:   return transform_erf(x, params.power_exp);
        case MAPPING_TANH:  return transform_tanh(x, params.power_exp);
        case MAPPING_EXP_STRETCH: return transform_exp_stretch(x, params.power_exp);
        default: return x;
    }
}

// ---- Linear bucketing for transform modes ----

__device__ __forceinline__ uint8_t linear_map_to_uint8(float val, float range_min, float inv_range) {
    int bin = __float2int_rd((val - range_min) * inv_range);
    return static_cast<uint8_t>(min(max(bin, 0), 255));
}

// ---- BF16-aware bucketing (mode 8) ----
// BF16 has 8 exponent + 7 mantissa bits.  Taking the upper 8 bits of the
// sign-flipped bf16 bit-pattern yields only ~20 distinct bins for typical
// data (the byte is almost entirely exponent).  Instead, convert through
// fp16 (5 exp + 10 mantissa) which puts 5 exp + 2 mantissa bits in the
// upper byte, giving ~135+ distinct bins — equivalent to mode 0 but
// explicitly available as a named mode for documentation/benchmarking.

__device__ __forceinline__ uint8_t convert_to_uint8_bf16(float x) {
    return convert_to_uint8(x);  // fp16 sign-flip bucketing
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
        case MAPPING_LOG1P:
        case MAPPING_ERF:
        case MAPPING_TANH:
        case MAPPING_EXP_STRETCH: {
            float val = apply_transform(x, params);
            if (params.noscale) return convert_to_uint8(val);
            return linear_map_to_uint8(val, range_min, inv_range);
        }
        case MAPPING_TRUNC8:
            return convert_to_uint8_bf16(x);
        case MAPPING_SUBTRACT:
            return convert_to_uint8(x - range_min);  // range_min repurposed as pivot
        case MAPPING_ADAPTIVE_TAIL_WINDOW:
        case MAPPING_TOPK_WINDOW:
            return linear_map_to_uint8(x, range_min, inv_range);
        default:  // MAPPING_NONE
            return convert_to_uint8(x);
    }
}

// Helper: check if a mapping mode needs the auto-range pre-pass
__device__ __forceinline__ bool needs_auto_range(int mode) {
    return (mode == MAPPING_POWER || mode == MAPPING_LOG ||
            mode == MAPPING_ASINH || mode == MAPPING_LOG1P ||
            mode == MAPPING_ERF || mode == MAPPING_TANH ||
            mode == MAPPING_EXP_STRETCH);
}

// Helper: check if a mapping mode needs the pivot pre-pass
__device__ __forceinline__ bool needs_pivot(int mode) {
    return (mode == MAPPING_SUBTRACT);
}

// Helper: check if mode is the adaptive tail-window pre-pass
__device__ __forceinline__ bool needs_tail_window(int mode) {
    return (mode == MAPPING_ADAPTIVE_TAIL_WINDOW);
}

// Helper: check if mode is the lightweight topk-window pre-pass
__device__ __forceinline__ bool needs_topk_window(int mode) {
    return (mode == MAPPING_TOPK_WINDOW);
}
