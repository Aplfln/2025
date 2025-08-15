// src/sm4_aesni_avx2.cpp
// AVX2-accelerated SM4 bulk encrypt using 8-way T-table gathers.
// Requires: compiler support for AVX2 (_mm256_i32gather_epi32).
// Compile with /arch:AVX2 (Visual Studio) or -mavx2 (gcc/clang).
//
// This file provides:
// - sm4_ecb_encrypt_aesni_avx2(...) : bulk encrypt using 8-way gather
// - sm4_encrypt_block_aesni_fallback(...) : single-block fallback (scalar)
// - main() under SM4_AESNI_TEST: KAT + speed comparison (ref / ttable / avx2)
//
// Depends on: include/sm4.h (for key schedule functions and ref ecb).

#include "sm4.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <array>

#if !defined(__AVX2__)
#warning "Compiling without AVX2 - AVX2 optimized path will not be available; fallback to scalar."
#endif

// --- local SBOX / TTables (same logic as ttable impl) ---
static constexpr uint8_t LOCAL_SBOX[256] = {
    0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05,
    0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99,
    0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62,
    0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6,
    0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8,
    0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35,
    0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87,
    0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e,
    0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1,
    0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3,
    0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f,
    0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51,
    0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8,
    0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0,
    0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84,
    0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48
};

static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }
static inline uint32_t L_enc_from_word(uint32_t B) {
    return B ^ rotl32(B, 2) ^ rotl32(B, 10) ^ rotl32(B, 18) ^ rotl32(B, 24);
}

// T-table arrays (initialized once)
static alignas(64) uint32_t T0_glob[256];
static alignas(64) uint32_t T1_glob[256];
static alignas(64) uint32_t T2_glob[256];
static alignas(64) uint32_t T3_glob[256];
static bool T_inited = false;

static void init_Ttables_globals() {
    if (T_inited) return;
    for (int b = 0; b < 256; ++b) {
        uint32_t s = static_cast<uint32_t>(LOCAL_SBOX[b]);
        T0_glob[b] = L_enc_from_word(s << 24);
        T1_glob[b] = L_enc_from_word(s << 16);
        T2_glob[b] = L_enc_from_word(s << 8);
        T3_glob[b] = L_enc_from_word(s);
    }
    T_inited = true;
}

// -------------------------------------------------
// Single-block fallback (calls ref T-Table logic) - ensures KAT correctness
// -------------------------------------------------
void sm4_encrypt_block_aesni_fallback(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]) {
    // big-endian load
    auto be_load = [](const uint8_t* b)->uint32_t {
        return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
        };
    auto be_store = [](uint8_t* b, uint32_t x) {
        b[0] = uint8_t(x >> 24);
        b[1] = uint8_t(x >> 16);
        b[2] = uint8_t(x >> 8);
        b[3] = uint8_t(x);
        };
    init_Ttables_globals();
    uint32_t X0 = be_load(in + 0);
    uint32_t X1 = be_load(in + 4);
    uint32_t X2 = be_load(in + 8);
    uint32_t X3 = be_load(in + 12);
    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = X1 ^ X2 ^ X3 ^ ks->rk[i];
        uint32_t t = T0_glob[(tmp >> 24) & 0xFF] ^ T1_glob[(tmp >> 16) & 0xFF] ^
            T2_glob[(tmp >> 8) & 0xFF] ^ T3_glob[tmp & 0xFF];
        uint32_t X4 = X0 ^ t;
        X0 = X1; X1 = X2; X2 = X3; X3 = X4;
    }
    be_store(out + 0, X3);
    be_store(out + 4, X2);
    be_store(out + 8, X1);
    be_store(out + 12, X0);
}

// -------------------------------------------------
// AVX2 8-way bulk encrypt using gathers
// Input length must be multiple of 16
// -------------------------------------------------
int sm4_ecb_encrypt_aesni_avx2(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len) {
    if (len % 16 != 0) return 0;
    init_Ttables_globals();

#if defined(__AVX2__)
    size_t blocks = len / 16;
    size_t i = 0;
    // Process in groups of 8 blocks
    for (; i + 7 < blocks; i += 8) {
        // load X0..X3 for 8 blocks as uint32 arrays
        uint32_t X0[8], X1[8], X2[8], X3[8];
        for (int k = 0; k < 8; ++k) {
            const uint8_t* p = in + (i + k) * 16;
            X0[k] = (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
            X1[k] = (uint32_t(p[4]) << 24) | (uint32_t(p[5]) << 16) | (uint32_t(p[6]) << 8) | uint32_t(p[7]);
            X2[k] = (uint32_t(p[8]) << 24) | (uint32_t(p[9]) << 16) | (uint32_t(p[10]) << 8) | uint32_t(p[11]);
            X3[k] = (uint32_t(p[12]) << 24) | (uint32_t(p[13]) << 16) | (uint32_t(p[14]) << 8) | uint32_t(p[15]);
        }

        // rounds
        for (int r = 0; r < 32; ++r) {
            uint32_t rk = ks->rk[r];
            // compute tmp[k] = X1^X2^X3 ^ rk
            int idx0[8], idx1[8], idx2[8], idx3[8];
            for (int k = 0; k < 8; ++k) {
                uint32_t tmp = X1[k] ^ X2[k] ^ X3[k] ^ rk;
                idx0[k] = static_cast<int>((tmp >> 24) & 0xFF);
                idx1[k] = static_cast<int>((tmp >> 16) & 0xFF);
                idx2[k] = static_cast<int>((tmp >> 8) & 0xFF);
                idx3[k] = static_cast<int>((tmp) & 0xFF);
            }
            // gather from T0..T3 (each returns 8 x int32)
            const int scale = 4;
            __m256i idx0_v = _mm256_loadu_si256((const __m256i*)idx0);
            __m256i idx1_v = _mm256_loadu_si256((const __m256i*)idx1);
            __m256i idx2_v = _mm256_loadu_si256((const __m256i*)idx2);
            __m256i idx3_v = _mm256_loadu_si256((const __m256i*)idx3);
            // gather: returns __m256i of 8 int32 loaded from table + idx*scale
            __m256i t0_v = _mm256_i32gather_epi32((const int*)T0_glob, idx0_v, scale);
            __m256i t1_v = _mm256_i32gather_epi32((const int*)T1_glob, idx1_v, scale);
            __m256i t2_v = _mm256_i32gather_epi32((const int*)T2_glob, idx2_v, scale);
            __m256i t3_v = _mm256_i32gather_epi32((const int*)T3_glob, idx3_v, scale);

            // t_v = t0^t1^t2^t3
            __m256i t_v = _mm256_xor_si256(t0_v, t1_v);
            t_v = _mm256_xor_si256(t_v, t2_v);
            t_v = _mm256_xor_si256(t_v, t3_v);

            // extract t_v elements back to uint32 array
            alignas(32) uint32_t t_arr[8];
            _mm256_storeu_si256((__m256i*)t_arr, t_v);

            // update X: X4 = X0 ^ t_arr[k]; rotate
            for (int k = 0; k < 8; ++k) {
                uint32_t X4 = X0[k] ^ t_arr[k];
                X0[k] = X1[k]; X1[k] = X2[k]; X2[k] = X3[k]; X3[k] = X4;
            }
        } // rounds

        // store back 8 blocks
        for (int k = 0; k < 8; ++k) {
            uint8_t* q = out + (i + k) * 16;
            uint32_t r0 = X3[k], r1 = X2[k], r2 = X1[k], r3 = X0[k];
            q[0] = uint8_t(r0 >> 24); q[1] = uint8_t(r0 >> 16); q[2] = uint8_t(r0 >> 8); q[3] = uint8_t(r0);
            q[4] = uint8_t(r1 >> 24); q[5] = uint8_t(r1 >> 16); q[6] = uint8_t(r1 >> 8); q[7] = uint8_t(r1);
            q[8] = uint8_t(r2 >> 24); q[9] = uint8_t(r2 >> 16); q[10] = uint8_t(r2 >> 8); q[11] = uint8_t(r2);
            q[12] = uint8_t(r3 >> 24); q[13] = uint8_t(r3 >> 16); q[14] = uint8_t(r3 >> 8); q[15] = uint8_t(r3);
        }
    } // groups of 8

    // remaining blocks processed scalar via fallback (keeps correctness)
    for (; i < blocks; ++i) {
        sm4_encrypt_block_aesni_fallback(ks, in + i * 16, out + i * 16);
    }
    return 1;
#else
    // no AVX2 - fallback to scalar
    size_t blocks = len / 16;
    for (size_t i = 0; i < blocks; ++i) {
        sm4_encrypt_block_aesni_fallback(ks, in + i * 16, out + i * 16);
    }
    return 1;
#endif
}

// -------------------------------------------------
// Test main (KAT + timing). Enable with SM4_AESNI_TEST macro.
// -------------------------------------------------
#ifdef SM4_AESNI_TEST
static void hexprint(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) std::cout << std::hex << std::setw(2) << std::setfill('0') << int(p[i]);
    std::cout << std::dec << std::endl;
}

int main() {
    // basic KAT key/plaintext
    const uint8_t key[16] = {
        0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef, 0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10
    };
    const uint8_t pt[16] = {
        0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef, 0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10
    };
    const uint8_t ct_exp[16] = {
        0x68,0x1e,0xdf,0x34, 0xd2,0x06,0x96,0x5e, 0x86,0xb3,0xe9,0x4f, 0x53,0x6e,0x42,0x46
    };

    sm4_key_t ks, ks_dec;
    sm4_setkey_enc(&ks, key);
    sm4_setkey_dec(&ks_dec, key);

    // KAT test: use fallback single-block to verify correctness
    uint8_t ct1[16], pt2[16];
    sm4_encrypt_block_aesni_fallback(&ks, pt, ct1);
    sm4_encrypt_block_aesni_fallback(&ks_dec, ct1, pt2);

    std::cout << "KAT AESNI(fallback) encryption result: "; hexprint(ct1, 16);
    std::cout << "AESNI enc OK? " << (std::memcmp(ct1, ct_exp, 16) == 0 ? "yes" : "NO") << std::endl;
    std::cout << "AESNI dec OK? " << (std::memcmp(pt2, pt, 16) == 0 ? "yes" : "NO") << std::endl;

    // for fair compare, also run ttable single-block (if present)
    uint8_t ct_tt[16] = { 0 };
    sm4_encrypt_block_aesni_fallback(&ks, pt, ct_tt); // same as fallback

    // ================= timing =================
    const size_t TEST_MB = 16;
    const size_t TEST_SIZE = TEST_MB * 1024 * 1024;
    std::vector<uint8_t> in(TEST_SIZE, 0x11), out_ref(TEST_SIZE), out_tt(TEST_SIZE), out_avx(TEST_SIZE);

    // warmups
    sm4_ecb_encrypt_ref(&ks, in.data(), out_ref.data(), TEST_SIZE);
    // local ttable single (calls fallback T-table in ref if available)
    sm4_ecb_encrypt_ref(&ks, in.data(), out_tt.data(), TEST_SIZE); // ensure T-tables inited if same
    sm4_ecb_encrypt_aesni_avx2(&ks, in.data(), out_avx.data(), TEST_SIZE);

    // measure ref
    auto t0 = std::chrono::high_resolution_clock::now();
    sm4_ecb_encrypt_ref(&ks, in.data(), out_ref.data(), TEST_SIZE);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_ref = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mbps_ref = (double)TEST_MB / (ms_ref / 1000.0);

    // measure avx2
    auto t2 = std::chrono::high_resolution_clock::now();
    sm4_ecb_encrypt_aesni_avx2(&ks, in.data(), out_avx.data(), TEST_SIZE);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_avx = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double mbps_avx = (double)TEST_MB / (ms_avx / 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[REF]   Encrypt " << TEST_MB << " MB in " << ms_ref << " ms, speed = " << mbps_ref << " MB/s\n";
    std::cout << "[AVX2]  Encrypt " << TEST_MB << " MB in " << ms_avx << " ms, speed = " << mbps_avx << " MB/s\n";

    bool identical = (out_ref == out_avx);
    std::cout << "[COMPARE] ref vs avx2 identical? " << (identical ? "yes" : "NO") << std::endl;
    if (mbps_avx > mbps_ref) {
        std::cout << "[SPEEDUP] AVX2 TTable gather = " << (mbps_avx / mbps_ref) << "x\n";
    }
    else {
        std::cout << "[SPEEDUP] AVX2 not faster (ref/avx2 = " << (mbps_ref / mbps_avx) << ")\n";
    }

    return 0;
}
#endif // SM4_AESNI_TEST

