#ifndef HASH_FUNCTIONS_CUH_
#define HASH_FUNCTIONS_CUH_

#include <cuda_runtime.h>
#include <stdint.h>

// xxHash64 constants (from OneFlow's implementation)
constexpr uint64_t PRIME64_1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t PRIME64_3 = 0x165667B19E3779F9ULL;
constexpr uint64_t PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t PRIME64_5 = 0x27D4EB2F165667C5ULL;

#define XXH_rotl64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))

__device__ inline uint64_t XXH64_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = XXH_rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

// xxHash64 implementation (based on OneFlow)
__device__ inline uint64_t xxh64_uint64(uint64_t v, uint64_t seed) {
    uint64_t acc = seed + PRIME64_5;
    acc += sizeof(uint64_t);
    acc = acc ^ XXH64_round(0, v);
    acc = XXH_rotl64(acc, 27) * PRIME64_1;
    acc = acc + PRIME64_4;
    acc ^= (acc >> 33);
    acc = acc * PRIME64_2;
    acc = acc ^ (acc >> 29);
    acc = acc * PRIME64_3;
    acc = acc ^ (acc >> 32);
    return acc;
}

// 30 diverse hash functions using xxHash64 with carefully chosen seeds for maximum independence
__device__ inline uint64_t hash_func_0(uint64_t x) {
    return xxh64_uint64(x, 0x243F6A8885A308D3ULL);  // First 16 hex digits of π
}

__device__ inline uint64_t hash_func_1(uint64_t x) {
    return xxh64_uint64(x, 0x13198A2E03707344ULL);  // Next 16 hex digits of π
}

__device__ inline uint64_t hash_func_2(uint64_t x) {
    return xxh64_uint64(x, 0xA4093822299F31D0ULL);  // Continue π digits
}

__device__ inline uint64_t hash_func_3(uint64_t x) {
    return xxh64_uint64(x, 0x082EFA98EC4E6C89ULL);  // Continue π digits
}

__device__ inline uint64_t hash_func_4(uint64_t x) {
    return xxh64_uint64(x, 0x452821E638D01377ULL);  // Continue π digits
}

__device__ inline uint64_t hash_func_5(uint64_t x) {
    return xxh64_uint64(x, 0xBE5466CF34E90C6CULL);  // First 16 hex digits of e
}

__device__ inline uint64_t hash_func_6(uint64_t x) {
    return xxh64_uint64(x, 0xC0AC29B7C97C50DDULL);  // Next 16 hex digits of e
}

__device__ inline uint64_t hash_func_7(uint64_t x) {
    return xxh64_uint64(x, 0x3F84D5B5B5470917ULL);  // Continue e digits
}

__device__ inline uint64_t hash_func_8(uint64_t x) {
    return xxh64_uint64(x, 0x9216D5D98979FB1BULL);  // Continue e digits
}

__device__ inline uint64_t hash_func_9(uint64_t x) {
    return xxh64_uint64(x, 0xD1310BA698DFB5ACULL);  // First 16 hex digits of √2
}

__device__ inline uint64_t hash_func_10(uint64_t x) {
    return xxh64_uint64(x, 0x2FFD72DBD01ADFB7ULL);  // Next 16 hex digits of √2
}

__device__ inline uint64_t hash_func_11(uint64_t x) {
    return xxh64_uint64(x, 0xB8E1AFED6A267E96ULL);  // Continue √2 digits
}

__device__ inline uint64_t hash_func_12(uint64_t x) {
    return xxh64_uint64(x, 0xBA7C9045F12C7F99ULL);  // First 16 hex digits of √3
}

__device__ inline uint64_t hash_func_13(uint64_t x) {
    return xxh64_uint64(x, 0x24A19947B3916CF7ULL);  // Next 16 hex digits of √3
}

__device__ inline uint64_t hash_func_14(uint64_t x) {
    return xxh64_uint64(x, 0x0801F2E2858EFC16ULL);  // Continue √3 digits
}

__device__ inline uint64_t hash_func_15(uint64_t x) {
    return xxh64_uint64(x, 0x636920D871574E69ULL);  // First 16 hex digits of √5
}

__device__ inline uint64_t hash_func_16(uint64_t x) {
    return xxh64_uint64(x, 0xA458FEA3F4933D7EULL);  // Next 16 hex digits of √5
}

__device__ inline uint64_t hash_func_17(uint64_t x) {
    return xxh64_uint64(x, 0x0D95748F728EB658ULL);  // Continue √5 digits
}

__device__ inline uint64_t hash_func_18(uint64_t x) {
    return xxh64_uint64(x, 0x718281828459045BULL);  // First 16 hex digits of ln(2)
}

__device__ inline uint64_t hash_func_19(uint64_t x) {
    return xxh64_uint64(x, 0x3F50A7B549E77C4FULL);  // Next 16 hex digits of ln(2)
}

__device__ inline uint64_t hash_func_20(uint64_t x) {
    return xxh64_uint64(x, 0x1A962CB58A3F83E2ULL);  // First 16 hex digits of golden ratio φ
}

__device__ inline uint64_t hash_func_21(uint64_t x) {
    return xxh64_uint64(x, 0xCF7B9A5E5E2A84B1ULL);  // Next 16 hex digits of φ
}

__device__ inline uint64_t hash_func_22(uint64_t x) {
    return xxh64_uint64(x, 0x5F3759DF77D85E2DULL);  // Magic number from fast inverse sqrt
}

__device__ inline uint64_t hash_func_23(uint64_t x) {
    return xxh64_uint64(x, 0xDEADBEEFCAFEBABEULL);  // Classic debug values
}

__device__ inline uint64_t hash_func_24(uint64_t x) {
    return xxh64_uint64(x, 0xFEEDFACEDEADC0DEULL);  // More debug classics
}

__device__ inline uint64_t hash_func_25(uint64_t x) {
    return xxh64_uint64(x, 0x0123456789ABCDEFUL);  // Sequential hex
}

__device__ inline uint64_t hash_func_26(uint64_t x) {
    return xxh64_uint64(x, 0xFEDCBA9876543210ULL);  // Reverse sequential
}

__device__ inline uint64_t hash_func_27(uint64_t x) {
    return xxh64_uint64(x, 0x9ABCDEF012345678ULL);  // Rotated sequential
}

__device__ inline uint64_t hash_func_28(uint64_t x) {
    return xxh64_uint64(x, 0xAAAAAAAAAAAAAAAAULL);  // Alternating bit pattern
}

__device__ inline uint64_t hash_func_29(uint64_t x) {
    return xxh64_uint64(x, 0x5555555555555555ULL);  // Inverse alternating pattern
}

__device__ inline uint64_t apply_hash(uint64_t x, int hash_id) {
    switch (hash_id) {
        case 0: return hash_func_0(x);
        case 1: return hash_func_1(x);
        case 2: return hash_func_2(x);
        case 3: return hash_func_3(x);
        case 4: return hash_func_4(x);
        case 5: return hash_func_5(x);
        case 6: return hash_func_6(x);
        case 7: return hash_func_7(x);
        case 8: return hash_func_8(x);
        case 9: return hash_func_9(x);
        case 10: return hash_func_10(x);
        case 11: return hash_func_11(x);
        case 12: return hash_func_12(x);
        case 13: return hash_func_13(x);
        case 14: return hash_func_14(x);
        case 15: return hash_func_15(x);
        case 16: return hash_func_16(x);
        case 17: return hash_func_17(x);
        case 18: return hash_func_18(x);
        case 19: return hash_func_19(x);
        case 20: return hash_func_20(x);
        case 21: return hash_func_21(x);
        case 22: return hash_func_22(x);
        case 23: return hash_func_23(x);
        case 24: return hash_func_24(x);
        case 25: return hash_func_25(x);
        case 26: return hash_func_26(x);
        case 27: return hash_func_27(x);
        case 28: return hash_func_28(x);
        case 29: return hash_func_29(x);
        default: return hash_func_0(x);
    }
}

#endif  // HASH_FUNCTIONS_CUH_
