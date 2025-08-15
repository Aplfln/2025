#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>
#include <chrono>

#if defined(_MSC_VER)
#include <intrin.h>    // __cpuid
#endif
#include <immintrin.h>   // AVX2 / PCLMUL / SSE

using std::uint8_t; using std::uint32_t; using std::uint64_t;

/*======================== SM4 基础参考实现（正确性基准） ========================*/
struct sm4_key_t { uint32_t rk[32]; };

static inline uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

static const uint32_t FK[4] = { 0xa3b1bac6u,0x56aa3350u,0x677d9197u,0xb27022dcu };
static const uint32_t CK[32] = {
    0x00070e15u,0x1c232a31u,0x383f464du,0x545b6269u,0x70777e85u,0x8c939aa1u,0xa8afb6bdu,0xc4cbd2d9u,
    0xe0e7eef5u,0xfc030a11u,0x181f262du,0x343b4249u,0x50575e65u,0x6c737a81u,0x888f969du,0xa4abb2b9u,
    0xc0c7ced5u,0xdce3eaf1u,0xf8ff060du,0x141b2229u,0x30373e45u,0x4c535a61u,0x686f767du,0x848b9299u,
    0xa0a7aeb5u,0xbcc3cad1u,0xd8dfe6edu,0xf4fb0209u,0x10171e25u,0x2c333a41u,0x484f565du,0x646b7279u
};
static const uint8_t SBOX[256] = {
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
static inline uint32_t LOAD_BE32(const uint8_t* p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3]; }
static inline void STORE_BE32(uint8_t* p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static inline uint32_t TAU(uint32_t x) {
    uint8_t b0 = SBOX[(x >> 24) & 0xff], b1 = SBOX[(x >> 16) & 0xff], b2 = SBOX[(x >> 8) & 0xff], b3 = SBOX[x & 0xff];
    return (uint32_t)b0 << 24 | (uint32_t)b1 << 16 | (uint32_t)b2 << 8 | (uint32_t)b3;
}
static inline uint32_t L_enc(uint32_t b) { return b ^ rotl32(b, 2) ^ rotl32(b, 10) ^ rotl32(b, 18) ^ rotl32(b, 24); }
static inline uint32_t L_key(uint32_t b) { return b ^ rotl32(b, 13) ^ rotl32(b, 23); }
static inline uint32_t T_enc(uint32_t x) { return L_enc(TAU(x)); }
static inline uint32_t T_key(uint32_t x) { return L_key(TAU(x)); }

static void sm4_setkey_enc(sm4_key_t* ks, const uint8_t key[16]) {
    uint32_t K0 = LOAD_BE32(key + 0) ^ FK[0], K1 = LOAD_BE32(key + 4) ^ FK[1], K2 = LOAD_BE32(key + 8) ^ FK[2], K3 = LOAD_BE32(key + 12) ^ FK[3];
    for (int i = 0; i < 32; i++) { uint32_t t = K1 ^ K2 ^ K3 ^ CK[i]; uint32_t rk = K0 ^ T_key(t); ks->rk[i] = rk; K0 = K1; K1 = K2; K2 = K3; K3 = rk; }
}
static void sm4_setkey_dec(sm4_key_t* kd, const uint8_t key[16]) { sm4_key_t ke; sm4_setkey_enc(&ke, key); for (int i = 0; i < 32; i++) kd->rk[i] = ke.rk[31 - i]; }

static void sm4_encrypt_block_ref(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]) {
    uint32_t X0 = LOAD_BE32(in + 0), X1 = LOAD_BE32(in + 4), X2 = LOAD_BE32(in + 8), X3 = LOAD_BE32(in + 12);
    for (int i = 0; i < 32; i++) { uint32_t tmp = X1 ^ X2 ^ X3 ^ ks->rk[i]; uint32_t X4 = X0 ^ T_enc(tmp); X0 = X1; X1 = X2; X2 = X3; X3 = X4; }
    STORE_BE32(out + 0, X3); STORE_BE32(out + 4, X2); STORE_BE32(out + 8, X1); STORE_BE32(out + 12, X0);
}
static void sm4_decrypt_block_ref(const sm4_key_t* ks, const uint8_t in[16], uint8_t out[16]) { sm4_encrypt_block_ref(ks, in, out); }

/*======================== 工具 / CPU 特性检测 ========================*/
static inline void store_be64(uint8_t* p, uint64_t v) { for (int i = 7; i >= 0; --i) p[7 - i] = (uint8_t)(v >> (i * 8)); }
static inline void gcm_inc32(uint8_t y[16]) {
    uint32_t c = ((uint32_t)y[12] << 24) | ((uint32_t)y[13] << 16) | ((uint32_t)y[14] << 8) | (uint32_t)y[15];
    c += 1; y[12] = (uint8_t)(c >> 24); y[13] = (uint8_t)(c >> 16); y[14] = (uint8_t)(c >> 8); y[15] = (uint8_t)c;
}
static inline bool cpu_supports_pclmul() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4]; __cpuid(info, 1); return (info[2] & (1 << 1)) != 0;
#else
    return true;
#endif
}
static inline bool cpu_supports_avx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int info[4]; __cpuid(info, 7); return (info[1] & (1 << 5)) != 0; // EBX bit5 = AVX2
#else
    return true;
#endif
}

/*======================== GHASH（PCLMUL 优化 + 标量回退） ========================*/
static inline void gf_shift_right_1(uint8_t y[16]) {
    uint8_t carry = 0; for (int i = 0; i < 16; i++) { uint8_t nc = y[15 - i] & 1; y[15 - i] = (y[15 - i] >> 1) | (carry ? 0x80 : 0); carry = nc; }
    if (carry) y[0] ^= 0xE1;
}
static inline void gf_mul_scalar(uint8_t x[16], const uint8_t h[16], uint8_t out[16]) {
    uint8_t Z[16] = { 0 }, V[16], X[16]; std::memcpy(V, h, 16); std::memcpy(X, x, 16);
    for (int i = 0; i < 128; i++) {
        if (X[15] & 1) for (int j = 0; j < 16; j++) Z[j] ^= V[j];
        uint8_t lsb = V[15] & 1, carry = 0;
        for (int j = 0; j < 16; j++) { uint8_t nc = V[j] & 1; V[j] = (V[j] >> 1) | (carry ? 0x80 : 0); carry = nc; }
        if (lsb) V[0] ^= 0xE1;
        gf_shift_right_1(X);
    }
    std::memcpy(out, Z, 16);
}
static inline __m128i bswap128(__m128i x) { const __m128i rev = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); return _mm_shuffle_epi8(x, rev); }
static inline __m128i ghash_mul_pclmul(__m128i X_be, __m128i H_be) {
    __m128i X = bswap128(X_be), H = bswap128(H_be);
    __m128i Z0 = _mm_clmulepi64_si128(X, H, 0x00);
    __m128i Z1 = _mm_xor_si128(_mm_clmulepi64_si128(X, H, 0x10), _mm_clmulepi64_si128(X, H, 0x01));
    __m128i Z2 = _mm_clmulepi64_si128(X, H, 0x11);
    __m128i V = _mm_xor_si128(Z0, _mm_slli_si128(Z1, 8));
    __m128i U = _mm_xor_si128(Z2, _mm_srli_si128(Z1, 8));
    __m128i T = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi64(U, 63), _mm_srli_epi64(U, 62)), _mm_srli_epi64(U, 57));
    V = _mm_xor_si128(V, T);
    __m128i U2 = _mm_xor_si128(_mm_xor_si128(_mm_slli_epi64(U, 1), _mm_slli_epi64(U, 2)), _mm_slli_epi64(U, 7));
    V = _mm_xor_si128(V, U2);
    return bswap128(V);
}
struct GHASH {
    uint8_t H[16];
    uint8_t Y[16];
    bool use_clmul;
};
static void ghash_init(GHASH& g, const uint8_t H[16]) { std::memcpy(g.H, H, 16); std::memset(g.Y, 0, 16); g.use_clmul = cpu_supports_pclmul(); }
static inline void ghash_block(GHASH& g, const uint8_t block[16]) {
    uint8_t X[16]; for (int i = 0; i < 16; i++) X[i] = g.Y[i] ^ block[i];
    if (g.use_clmul) {
        __m128i Xi = _mm_loadu_si128((const __m128i*)X), Hi = _mm_loadu_si128((const __m128i*)g.H);
        __m128i Zi = ghash_mul_pclmul(Xi, Hi); _mm_storeu_si128((__m128i*)g.Y, Zi);
    }
    else {
        uint8_t Z[16]; gf_mul_scalar(X, g.H, Z); std::memcpy(g.Y, Z, 16);
    }
}
static void ghash_update(GHASH& g, const uint8_t* data, size_t len) {
    while (len >= 16) { ghash_block(g, data); data += 16; len -= 16; }
    if (len) { uint8_t tmp[16] = { 0 }; std::memcpy(tmp, data, len); ghash_block(g, tmp); }
}

/*======================== CTR：AVX2 8-way 并行 + 标量回退 ========================*/

// 标量 CTR（回退/尾部处理）
static void sm4_ctr_crypt_scalar(const sm4_key_t* ks, const uint8_t* ctr0, const uint8_t* in, uint8_t* out, size_t len) {
    uint8_t ctr[16]; std::memcpy(ctr, ctr0, 16);
    size_t n = len / 16;
    for (size_t i = 0; i < n; i++) {
        uint8_t ksblk[16]; sm4_encrypt_block_ref(ks, ctr, ksblk);
        for (int b = 0; b < 16; b++) out[i * 16 + b] = in[i * 16 + b] ^ ksblk[b];
        gcm_inc32(ctr);
    }
    size_t rem = len & 15;
    if (rem) {
        uint8_t ksblk[16]; sm4_encrypt_block_ref(ks, ctr, ksblk);
        for (size_t b = 0; b < rem; b++) out[n * 16 + b] = in[n * 16 + b] ^ ksblk[b];
    }
}

// AVX2 8-way：每次处理 8*16=128 字节（生成 8 个计数器块的密钥流，用 AVX2 进行 256-bit XOR）
static void sm4_ctr_crypt_avx2_8way(const sm4_key_t* ks, const uint8_t* ctr0, const uint8_t* in, uint8_t* out, size_t len) {
    if (len < 128) { sm4_ctr_crypt_scalar(ks, ctr0, in, out, len); return; }

    uint8_t ctr[16]; std::memcpy(ctr, ctr0, 16);
    size_t nblk128 = len / 128;
    size_t remain = len % 128;

    // 预先准备 8 个计数器的副本
    uint8_t ctrs[8][16];
    uint8_t ksbuf[128];

    for (size_t blk = 0; blk < nblk128; ++blk) {
        // 准备 8 个连续计数器
        std::memcpy(ctrs[0], ctr, 16);
        for (int i = 1; i < 8; i++) { std::memcpy(ctrs[i], ctrs[i - 1], 16); gcm_inc32(ctrs[i]); }
        // 下一轮计数器起点（+8）
        for (int i = 0; i < 8; i++) gcm_inc32(ctr);

        // 生成 8 个 keystream 块（仍用标量 SM4；你可以在这里替换为更快的 T-table/AVX2 SM4）
        for (int i = 0; i < 8; i++) sm4_encrypt_block_ref(ks, ctrs[i], ksbuf + i * 16);

        // 用 AVX2 做 4 次 256-bit XOR（每次 32 字节 = 2 块）
        const __m256i* pin = (const __m256i*)(in + blk * 128);
        const __m256i* pks = (const __m256i*)(ksbuf);
        __m256i* pout = (__m256i*)(out + blk * 128);

        // 128B = 4 * 32B
        __m256i x0 = _mm256_loadu_si256(pin + 0);
        __m256i k0 = _mm256_loadu_si256(pks + 0);
        _mm256_storeu_si256(pout + 0, _mm256_xor_si256(x0, k0));

        __m256i x1 = _mm256_loadu_si256(pin + 1);
        __m256i k1 = _mm256_loadu_si256(pks + 1);
        _mm256_storeu_si256(pout + 1, _mm256_xor_si256(x1, k1));

        __m256i x2 = _mm256_loadu_si256(pin + 2);
        __m256i k2 = _mm256_loadu_si256(pks + 2);
        _mm256_storeu_si256(pout + 2, _mm256_xor_si256(x2, k2));

        __m256i x3 = _mm256_loadu_si256(pin + 3);
        __m256i k3 = _mm256_loadu_si256(pks + 3);
        _mm256_storeu_si256(pout + 3, _mm256_xor_si256(x3, k3));
    }

    if (remain) {
        // 处理剩余部分：从 ctr0 + 128*nblk 的计数器继续
        // 先把 ctr 推进到正确位置
        uint8_t ctr_tail[16]; std::memcpy(ctr_tail, ctr0, 16);
        // 推进 (nblk128 * 8) 次
        for (size_t i = 0; i < nblk128 * 8; i++) gcm_inc32(ctr_tail);
        sm4_ctr_crypt_scalar(ks, ctr_tail, in + nblk128 * 128, out + nblk128 * 128, remain);
    }
}

/*======================== SM4-GCM 顶层 ========================*/
struct SM4_GCM_CTX { sm4_key_t ks_enc; uint8_t H[16]; uint8_t J0[16]; };

static void sm4_gcm_setkey(SM4_GCM_CTX& c, const uint8_t key[16]) { sm4_setkey_enc(&c.ks_enc, key); uint8_t z[16] = { 0 }; sm4_encrypt_block_ref(&c.ks_enc, z, c.H); }
static void sm4_gcm_setiv(SM4_GCM_CTX& c, const uint8_t* iv, size_t iv_len) {
    if (iv_len == 12) { std::memcpy(c.J0, iv, 12); c.J0[12] = 0; c.J0[13] = 0; c.J0[14] = 0; c.J0[15] = 1; }
    else {
        GHASH g; ghash_init(g, c.H); ghash_update(g, iv, iv_len);
        uint8_t lenblk[16] = { 0 }; store_be64(lenblk + 8, (uint64_t)iv_len * 8); ghash_block(g, lenblk);
        std::memcpy(c.J0, g.Y, 16);
    }
}

static void sm4_gcm_encrypt(const uint8_t key[16],
    const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    const uint8_t* pt, size_t pt_len,
    uint8_t* ct, uint8_t tag[16]) {
    SM4_GCM_CTX ctx; sm4_gcm_setkey(ctx, key); sm4_gcm_setiv(ctx, iv, iv_len);

    GHASH g; ghash_init(g, ctx.H);
    if (aad_len) ghash_update(g, aad, aad_len);

    // CTR from J0+1
    uint8_t J0p1[16]; std::memcpy(J0p1, ctx.J0, 16); gcm_inc32(J0p1);
    if (cpu_supports_avx2()) sm4_ctr_crypt_avx2_8way(&ctx.ks_enc, J0p1, pt, ct, pt_len);
    else                     sm4_ctr_crypt_scalar(&ctx.ks_enc, J0p1, pt, ct, pt_len);

    if (pt_len) ghash_update(g, ct, pt_len);

    uint8_t lenblk[16] = { 0 }; store_be64(lenblk, (uint64_t)aad_len * 8); store_be64(lenblk + 8, (uint64_t)pt_len * 8); ghash_block(g, lenblk);

    uint8_t EKJ0[16]; sm4_encrypt_block_ref(&ctx.ks_enc, ctx.J0, EKJ0);
    for (int i = 0; i < 16; i++) tag[i] = EKJ0[i] ^ g.Y[i];
}

static bool sm4_gcm_decrypt_and_verify(const uint8_t key[16],
    const uint8_t* iv, size_t iv_len,
    const uint8_t* aad, size_t aad_len,
    const uint8_t* ct, size_t ct_len,
    const uint8_t tag[16],
    uint8_t* pt_out) {
    SM4_GCM_CTX ctx; sm4_gcm_setkey(ctx, key); sm4_gcm_setiv(ctx, iv, iv_len);

    GHASH g; ghash_init(g, ctx.H);
    if (aad_len) ghash_update(g, aad, aad_len);
    if (ct_len)  ghash_update(g, ct, ct_len);

    uint8_t lenblk[16] = { 0 }; store_be64(lenblk, (uint64_t)aad_len * 8); store_be64(lenblk + 8, (uint64_t)ct_len * 8); ghash_block(g, lenblk);

    uint8_t EKJ0[16]; sm4_encrypt_block_ref(&ctx.ks_enc, ctx.J0, EKJ0);
    uint8_t tag_calc[16]; for (int i = 0; i < 16; i++) tag_calc[i] = EKJ0[i] ^ g.Y[i];

    // CTR decrypt
    uint8_t J0p1[16]; std::memcpy(J0p1, ctx.J0, 16); gcm_inc32(J0p1);
    if (cpu_supports_avx2()) sm4_ctr_crypt_avx2_8way(&ctx.ks_enc, J0p1, ct, pt_out, ct_len);
    else                     sm4_ctr_crypt_scalar(&ctx.ks_enc, J0p1, ct, pt_out, ct_len);

    return std::memcmp(tag, tag_calc, 16) == 0;
}

/*======================== Demo / Benchmark main ========================*/
static void hexprint(const uint8_t* p, size_t n) { for (size_t i = 0; i < n; i++) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)p[i]; std::cout << std::dec << std::endl; }

int main() {
    std::cout << "=== Single-file SM4-GCM (AVX2 CTR 8-way + PCLMUL GHASH) ===\n";

    // SM4 KAT
    const uint8_t sm4_key_kat[16] = { 0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef, 0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10 };
    const uint8_t sm4_pt_kat[16] = { 0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef, 0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10 };
    const uint8_t sm4_ct_exp[16] = { 0x68,0x1e,0xdf,0x34, 0xd2,0x06,0x96,0x5e, 0x86,0xb3,0xe9,0x4f, 0x53,0x6e,0x42,0x46 };

    sm4_key_t ks_e, ks_d; sm4_setkey_enc(&ks_e, sm4_key_kat); sm4_setkey_dec(&ks_d, sm4_key_kat);
    uint8_t ct_kat[16], pt_kat2[16];
    sm4_encrypt_block_ref(&ks_e, sm4_pt_kat, ct_kat);
    sm4_decrypt_block_ref(&ks_d, ct_kat, pt_kat2);

    std::cout << "SM4 KAT enc: "; hexprint(ct_kat, 16);
    std::cout << "SM4 enc OK? " << (std::memcmp(ct_kat, sm4_ct_exp, 16) == 0 ? "yes" : "NO") << "\n";
    std::cout << "SM4 dec OK? " << (std::memcmp(pt_kat2, sm4_pt_kat, 16) == 0 ? "yes" : "NO") << "\n\n";

    // GCM 自检
    const uint8_t key[16] = { 0x00,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    const uint8_t iv[12] = { 0xa1,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xab,0xac };
    const char* aad_str = "GCM-AAD"; const size_t aad_len = 7;

    const size_t PT_LEN = 64;
    uint8_t pt[PT_LEN], ct[PT_LEN], dec[PT_LEN], tag[16];
    for (size_t i = 0; i < PT_LEN; i++) pt[i] = (uint8_t)i;

    sm4_gcm_encrypt(key, iv, sizeof(iv), (const uint8_t*)aad_str, aad_len, pt, PT_LEN, ct, tag);
    bool ok = sm4_gcm_decrypt_and_verify(key, iv, sizeof(iv), (const uint8_t*)aad_str, aad_len, ct, PT_LEN, tag, dec);
    std::cout << "GCM self-check verify: " << (ok ? "OK" : "FAIL") << "\n";
    std::cout << "Tag: "; hexprint(tag, 16);

    // ====== 计时（16 MB）======
    const size_t TEST_MB = 16;
    const size_t TEST_SIZE = TEST_MB * 1024 * 1024;

    std::vector<uint8_t> buf_in(TEST_SIZE), buf_out(TEST_SIZE), buf_dec(TEST_SIZE);
    std::vector<uint8_t> AAD(1024, 0xA5); // 1KB AAD
    for (size_t i = 0; i < TEST_SIZE; i++) buf_in[i] = (uint8_t)i;

    // 预热
    uint8_t tag_tmp[16];
    sm4_gcm_encrypt(key, iv, sizeof(iv), AAD.data(), AAD.size(), buf_in.data(), TEST_SIZE, buf_out.data(), tag_tmp);

    // 加密计时
    auto t1 = std::chrono::high_resolution_clock::now();
    sm4_gcm_encrypt(key, iv, sizeof(iv), AAD.data(), AAD.size(), buf_in.data(), TEST_SIZE, buf_out.data(), tag_tmp);
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_enc = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double mbps = TEST_MB / (ms_enc / 1000.0);

    // 解密+验签计时
    auto t3 = std::chrono::high_resolution_clock::now();
    bool ok2 = sm4_gcm_decrypt_and_verify(key, iv, sizeof(iv), AAD.data(), AAD.size(), buf_out.data(), TEST_SIZE, tag_tmp, buf_dec.data());
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms_dec = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double mbps_dec = TEST_MB / (ms_dec / 1000.0);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[GCM-ENC] Encrypt " << TEST_MB << " MB in " << ms_enc << " ms, speed = " << mbps << " MB/s\n";
    std::cout << "[GCM-DEC] Decrypt+Verify " << TEST_MB << " MB in " << ms_dec << " ms, speed = " << mbps_dec << " MB/s\n";
    std::cout << "[VERIFY] " << (ok2 ? "PASS" : "FAIL") << "\n";

    // 额外提示：是否走了 AVX2 / PCLMUL
    std::cout << "[PATH] AVX2=" << (cpu_supports_avx2() ? "on" : "off")
        << ", PCLMUL=" << (cpu_supports_pclmul() ? "on" : "off") << "\n";
    return (ok && ok2) ? 0 : 1;
}
