// src/sm4_ttable.cpp
// SM4 T-Table implementation + 4-way interleaved encrypt + timing test
// Usage: compile together with sm4_ref.cpp and include/sm4.h
// Define SM4_TTABLE_TEST to compile test main.
//
// Example (g++):
// g++ -O3 -std=c++17 src/sm4_ref.cpp src/sm4_ttable.cpp -DSM4_TTABLE_TEST -o sm4_ttable_test
//
// In Visual Studio: add this file to project and add SM4_TTABLE_TEST to preprocessor defs for Debug/Release as needed.

#include "sm4.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

// Local SBOX (same as in ref) used to init tables
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

// utility: rotl32 (same as ref)
static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

// We'll implement T_enc as L_enc(SBOX(byte) placed in position) during table init
static inline uint32_t L_enc_from_word(uint32_t B) {
    return B ^ rotl32(B, 2) ^ rotl32(B, 10) ^ rotl32(B, 18) ^ rotl32(B, 24);
}

struct TTables {
    std::array<uint32_t, 256> T0, T1, T2, T3;
    void init() {
        for (int b = 0; b < 256; ++b) {
            uint32_t s = static_cast<uint32_t>(LOCAL_SBOX[b]);
            T0[b] = L_enc_from_word(s << 24);
            T1[b] = L_enc_from_word(s << 16);
            T2[b] = L_enc_from_word(s << 8);
            T3[b] = L_enc_from_word(s);
        }
    }
};

// single-block T-Table encrypt, overrides placeholder in sm4_ref.cpp
void sm4_encrypt_block_ttable(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]) {
    // load big-endian words
    auto be_load = [](const uint8_t* b)->uint32_t {
        return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
        };
    auto be_store = [](uint8_t* b, uint32_t x) {
        b[0] = uint8_t(x >> 24);
        b[1] = uint8_t(x >> 16);
        b[2] = uint8_t(x >> 8);
        b[3] = uint8_t(x);
        };

    // static table (init once)
    static TTables TT;
    static bool initialized = false;
    if (!initialized) { TT.init(); initialized = true; }

    uint32_t X0 = be_load(in + 0);
    uint32_t X1 = be_load(in + 4);
    uint32_t X2 = be_load(in + 8);
    uint32_t X3 = be_load(in + 12);

    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = X1 ^ X2 ^ X3 ^ ks->rk[i];
        uint32_t t = TT.T0[(tmp >> 24) & 0xFF] ^ TT.T1[(tmp >> 16) & 0xFF] ^
            TT.T2[(tmp >> 8) & 0xFF] ^ TT.T3[(tmp) & 0xFF];
        uint32_t X4 = X0 ^ t;
        X0 = X1; X1 = X2; X2 = X3; X3 = X4;
    }
    be_store(out + 0, X3);
    be_store(out + 4, X2);
    be_store(out + 8, X1);
    be_store(out + 12, X0);
}

// simple ECB helper using ttable
static int sm4_ecb_encrypt_ttable_local(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len) {
    if (len % 16 != 0) return 0;
    for (size_t i = 0; i < len; i += 16) sm4_encrypt_block_ttable(ks, in + i, out + i);
    return 1;
}

// 4-way interleaved encrypt for throughput: processes 4 blocks per iteration
// input/out pointers must have at least 4*16 bytes for each group; len in bytes
static int sm4_ecb_encrypt_ttable_4way(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len) {
    if (len % 16 != 0) return 0;
    size_t blocks = len / 16;
    size_t i = 0;
    // init TT locally (cheap due to static init in block function as well)
    static TTables TT;
    static bool initialized = false;
    if (!initialized) { TT.init(); initialized = true; }

    auto be_load = [](const uint8_t* b)->uint32_t {
        return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
        };
    auto be_store = [](uint8_t* b, uint32_t x) {
        b[0] = uint8_t(x >> 24);
        b[1] = uint8_t(x >> 16);
        b[2] = uint8_t(x >> 8);
        b[3] = uint8_t(x);
        };

    // process groups of 4 blocks
    for (; i + 3 < blocks; i += 4) {
        // load 4 blocks' state
        uint32_t X0[4], X1[4], X2[4], X3[4];
        for (int k = 0; k < 4; ++k) {
            const uint8_t* p = in + (i + k) * 16;
            X0[k] = be_load(p + 0);
            X1[k] = be_load(p + 4);
            X2[k] = be_load(p + 8);
            X3[k] = be_load(p + 12);
        }
        // 32 rounds
        for (int r = 0; r < 32; ++r) {
            uint32_t rk = ks->rk[r];
            for (int k = 0; k < 4; ++k) {
                uint32_t tmp = X1[k] ^ X2[k] ^ X3[k] ^ rk;
                uint32_t t = TT.T0[(tmp >> 24) & 0xFF] ^ TT.T1[(tmp >> 16) & 0xFF] ^
                    TT.T2[(tmp >> 8) & 0xFF] ^ TT.T3[(tmp) & 0xFF];
                uint32_t X4 = X0[k] ^ t;
                X0[k] = X1[k]; X1[k] = X2[k]; X2[k] = X3[k]; X3[k] = X4;
            }
        }
        // store back
        for (int k = 0; k < 4; ++k) {
            uint8_t* q = out + (i + k) * 16;
            be_store(q + 0, X3[k]);
            be_store(q + 4, X2[k]);
            be_store(q + 8, X1[k]);
            be_store(q + 12, X0[k]);
        }
    }
    // remaining blocks
    for (; i < blocks; ++i) {
        sm4_encrypt_block_ttable(ks, in + i * 16, out + i * 16);
    }
    return 1;
}

#ifdef SM4_TTABLE_TEST
// ---------------- test main with timing ----------------
static void hexprint(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i) std::cout << std::hex << std::setw(2) << std::setfill('0') << int(p[i]);
    std::cout << std::dec << std::endl;
}

int main() {
    // KAT vectors
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

    uint8_t ct_ref[16], pt2_ref[16];
    sm4_encrypt_block_ref(&ks, pt, ct_ref);
    sm4_encrypt_block_ref(&ks_dec, ct_ref, pt2_ref);

    std::cout << "KAT reference encryption result: "; hexprint(ct_ref, 16);
    std::cout << "Ref enc OK? " << (std::memcmp(ct_ref, ct_exp, 16) == 0 ? "yes" : "NO") << std::endl;
    std::cout << "Ref dec OK? " << (std::memcmp(pt2_ref, pt, 16) == 0 ? "yes" : "NO") << std::endl;

    // T-Table single-block KAT
    uint8_t ct_tt[16], pt2_tt[16];
    sm4_encrypt_block_ttable(&ks, pt, ct_tt);
    sm4_encrypt_block_ttable(&ks_dec, ct_tt, pt2_tt);
    std::cout << "KAT T-Table encryption result: "; hexprint(ct_tt, 16);
    std::cout << "TT enc OK? " << (std::memcmp(ct_tt, ct_exp, 16) == 0 ? "yes" : "NO") << std::endl;
    std::cout << "TT dec OK? " << (std::memcmp(pt2_tt, pt, 16) == 0 ? "yes" : "NO") << std::endl;

    // ================= timing =================
    const size_t TEST_MB = 16;
    const size_t TEST_SIZE = TEST_MB * 1024 * 1024;
    std::vector<uint8_t> in(TEST_SIZE, 0x11), out_ref(TEST_SIZE), out_tt(TEST_SIZE), out_tt4(TEST_SIZE);

    // warmups
    sm4_ecb_encrypt_ref(&ks, in.data(), out_ref.data(), TEST_SIZE);
    sm4_ecb_encrypt_ttable_local(&ks, in.data(), out_tt.data(), TEST_SIZE);
    sm4_ecb_encrypt_ttable_4way(&ks, in.data(), out_tt4.data(), TEST_SIZE);

    // measure ref
    auto t0 = std::chrono::high_resolution_clock::now();
    sm4_ecb_encrypt_ref(&ks, in.data(), out_ref.data(), TEST_SIZE);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_ref = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double mbps_ref = (double)TEST_MB / (ms_ref / 1000.0);

    // measure ttable single
    auto t2 = std::chrono::high_resolution_clock::now();
    sm4_ecb_encrypt_ttable_local(&ks, in.data(), out_tt.data(), TEST_SIZE);
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_tt = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double mbps_tt = (double)TEST_MB / (ms_tt / 1000.0);

    // measure ttable 4way
    auto t4 = std::chrono::high_resolution_clock::now();
    sm4_ecb_encrypt_ttable_4way(&ks, in.data(), out_tt4.data(), TEST_SIZE);
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_tt4 = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double mbps_tt4 = (double)TEST_MB / (ms_tt4 / 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[REF]    Encrypt " << TEST_MB << " MB in " << ms_ref << " ms, speed = " << mbps_ref << " MB/s\n";
    std::cout << "[TTABLE] Encrypt " << TEST_MB << " MB in " << ms_tt << " ms, speed = " << mbps_tt << " MB/s\n";
    std::cout << "[TT4WAY] Encrypt " << TEST_MB << " MB in " << ms_tt4 << " ms, speed = " << mbps_tt4 << " MB/s\n";

    // verify identical outputs
    bool id1 = (out_ref == out_tt);
    bool id2 = (out_ref == out_tt4);
    std::cout << "[COMPARE] ref vs ttable identical? " << (id1 ? "yes" : "NO") << std::endl;
    std::cout << "[COMPARE] ref vs ttable4 identical? " << (id2 ? "yes" : "NO") << std::endl;

    // print speedups
    if (mbps_tt > mbps_ref) {
        std::cout << "[SPEEDUP] TTable / ref = " << (mbps_tt / mbps_ref) << "x\n";
    }
    if (mbps_tt4 > mbps_ref) {
        std::cout << "[SPEEDUP] TTable-4way / ref = " << (mbps_tt4 / mbps_ref) << "x\n";
    }

    return 0;
}
*/
#endif // SM4_TTABLE_TEST
