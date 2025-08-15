// src/sm4_ref.cpp
// Reference SM4 implementation (C++). Provides key schedule, block encrypt, ECB helpers.
// Compile as: g++ -O3 -std=c++17 src/sm4_ref.cpp -o sm4_ref_test
#include "sm4.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

static constexpr uint32_t FK[4] = {
    0xa3b1bac6u, 0x56aa3350u, 0x677d9197u, 0xb27022dcu
};
static constexpr uint32_t CK[32] = {
    0x00070e15u, 0x1c232a31u, 0x383f464du, 0x545b6269u,
    0x70777e85u, 0x8c939aa1u, 0xa8afb6bdu, 0xc4cbd2d9u,
    0xe0e7eef5u, 0xfc030a11u, 0x181f262du, 0x343b4249u,
    0x50575e65u, 0x6c737a81u, 0x888f969du, 0xa4abb2b9u,
    0xc0c7ced5u, 0xdce3eaf1u, 0xf8ff060du, 0x141b2229u,
    0x30373e45u, 0x4c535a61u, 0x686f767du, 0x848b9299u,
    0xa0a7aeb5u, 0xbcc3cad1u, 0xd8dfe6edu, 0xf4fb0209u,
    0x10171e25u, 0x2c333a41u, 0x484f565du, 0x646b7279u
};

static constexpr uint8_t SBOX[256] = {
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
static inline uint32_t be_load_u32(const uint8_t b[4]) {
    return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | uint32_t(b[3]);
}
static inline void be_store_u32(uint8_t b[4], uint32_t x) {
    b[0] = uint8_t(x >> 24);
    b[1] = uint8_t(x >> 16);
    b[2] = uint8_t(x >> 8);
    b[3] = uint8_t(x);
}

static inline uint32_t tau(uint32_t x) {
    return (uint32_t(SBOX[(x >> 24) & 0xff]) << 24) |
        (uint32_t(SBOX[(x >> 16) & 0xff]) << 16) |
        (uint32_t(SBOX[(x >> 8) & 0xff]) << 8) |
        (uint32_t(SBOX[(x) & 0xff]));
}

static inline uint32_t L_enc(uint32_t B) {
    return B ^ rotl32(B, 2) ^ rotl32(B, 10) ^ rotl32(B, 18) ^ rotl32(B, 24);
}
static inline uint32_t L_key(uint32_t B) {
    return B ^ rotl32(B, 13) ^ rotl32(B, 23);
}
static inline uint32_t T_enc(uint32_t x) { return L_enc(tau(x)); }
static inline uint32_t T_key(uint32_t x) { return L_key(tau(x)); }

void sm4_setkey_enc(sm4_key_t* ks, const uint8_t key[16]) {
    uint32_t MK[4] = {
        be_load_u32(key + 0), be_load_u32(key + 4),
        be_load_u32(key + 8), be_load_u32(key + 12)
    };
    uint32_t K[4] = { MK[0] ^ FK[0], MK[1] ^ FK[1], MK[2] ^ FK[2], MK[3] ^ FK[3] };
    for (int i = 0; i < 32; ++i) {
        uint32_t t = K[1] ^ K[2] ^ K[3] ^ CK[i];
        uint32_t rk = K[0] ^ T_key(t);
        ks->rk[i] = rk;
        K[0] = K[1]; K[1] = K[2]; K[2] = K[3]; K[3] = rk;
    }
}

void sm4_setkey_dec(sm4_key_t* ks_dec, const uint8_t key[16]) {
    sm4_key_t tmp{};
    sm4_setkey_enc(&tmp, key);
    for (int i = 0; i < 32; ++i) ks_dec->rk[i] = tmp.rk[31 - i];
}

void sm4_encrypt_block_ref(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]) {
    uint32_t X0 = be_load_u32(in + 0);
    uint32_t X1 = be_load_u32(in + 4);
    uint32_t X2 = be_load_u32(in + 8);
    uint32_t X3 = be_load_u32(in + 12);
    for (int i = 0; i < 32; ++i) {
        uint32_t tmp = X1 ^ X2 ^ X3 ^ ks->rk[i];
        uint32_t X4 = X0 ^ T_enc(tmp);
        X0 = X1; X1 = X2; X2 = X3; X3 = X4;
    }
    be_store_u32(out + 0, X3);
    be_store_u32(out + 4, X2);
    be_store_u32(out + 8, X1);
    be_store_u32(out + 12, X0);
}

int sm4_ecb_encrypt_ref(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len) {
    if (len % SM4_BLOCK_BYTES != 0) return 0;
    for (size_t i = 0; i < len; i += SM4_BLOCK_BYTES) sm4_encrypt_block_ref(ks, in + i, out + i);
    return 1;
}
int sm4_ecb_decrypt_ref(const sm4_key_t* ks, const uint8_t* in, uint8_t* out, size_t len) {
    if (len % SM4_BLOCK_BYTES != 0) return 0;
    // decryption uses reversed round keys; user should call sm4_setkey_dec to prepare ks_dec
    for (size_t i = 0; i < len; i += SM4_BLOCK_BYTES) sm4_encrypt_block_ref(ks, in + i, out + i);
    return 1;
}


// stub placeholders for other backends (implemented later)



// ----------------- optional test main -----------------
/*
#ifdef SM4_REF_TEST
#include <chrono>
#include <vector>

static void hexprint(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; ++i)
        std::cout << std::hex << std::setw(2) << std::setfill('0') << int(p[i]);
    std::cout << std::dec << std::endl;
}

int main() {
    const uint8_t key[16] = {
        0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef,
        0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10
    };
    const uint8_t pt[16] = {
        0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef,
        0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10
    };
    const uint8_t ct_exp[16] = {
        0x68,0x1e,0xdf,0x34, 0xd2,0x06,0x96,0x5e,
        0x86,0xb3,0xe9,0x4f, 0x53,0x6e,0x42,0x46
    };

    sm4_key_t ks;
    sm4_setkey_enc(&ks, key);
    uint8_t ct[16], pt2[16];
    sm4_encrypt_block_ref(&ks, pt, ct);

    sm4_key_t ks_dec;
    sm4_setkey_dec(&ks_dec, key);
    sm4_encrypt_block_ref(&ks_dec, ct, pt2);

    std::cout << "CT: "; hexprint(ct, 16);
    std::cout << "enc ok? " << (std::memcmp(ct, ct_exp, 16) == 0 ? "yes" : "NO") << std::endl;
    std::cout << "dec ok? " << (std::memcmp(pt2, pt, 16) == 0 ? "yes" : "NO") << std::endl;

    // ================== ÐÔÄÜ²âÊÔ ==================
    const size_t TEST_MB = 16; // ²âÊÔ 16MB
    const size_t TEST_SIZE = TEST_MB * 1024 * 1024;
    std::vector<uint8_t> buf_in(TEST_SIZE, 0xAA);
    std::vector<uint8_t> buf_out(TEST_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    sm4_ecb_encrypt_ref(&ks, buf_in.data(), buf_out.data(), TEST_SIZE);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = end - start;
    double speed = TEST_MB / (ms.count() / 1000.0);

    std::cout << "[REF] Encrypt " << TEST_MB << " MB in "
        << ms.count() << " ms, speed = "
        << speed << " MB/s" << std::endl;

    return 0;
}
#endif
*/