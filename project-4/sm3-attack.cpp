#include <iostream>
#include <vector>
#include <cstring>
#include <iomanip>
#include <stdint.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ---------------- SM3 基本定义 ----------------
#define ROTL(x,n) (((x) << (n)) | ((x) >> (32-(n))))
#define FF0(x,y,z) ((x) ^ (y) ^ (z))
#define FF1(x,y,z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG0(x,y,z) ((x) ^ (y) ^ (z))
#define GG1(x,y,z) (((x) & (y)) | ((~(x)) & (z)))
#define P0(x) ((x) ^ ROTL((x),9) ^ ROTL((x),17))
#define P1(x) ((x) ^ ROTL((x),15) ^ ROTL((x),23))

const uint32_t IV[8] = {
    0x7380166f,
    0x4914b2b9,
    0x172442d7,
    0xda8a0600,
    0xa96f30bc,
    0x163138aa,
    0xe38dee4d,
    0xb0fb0e4e
};

const uint32_t T_j[64] = {
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a
};

// ---------------- SM3 压缩函数 ----------------
void sm3_compress(uint32_t V[8], const uint8_t B[64]) {
    uint32_t W[68], W1[64];
    for (int i = 0; i < 16; i++) {
        W[i] = (B[4 * i] << 24) | (B[4 * i + 1] << 16) | (B[4 * i + 2] << 8) | B[4 * i + 3];
    }
    for (int i = 16; i < 68; i++) {
        W[i] = P1(W[i - 16] ^ W[i - 9] ^ ROTL(W[i - 3], 15)) ^ ROTL(W[i - 13], 7) ^ W[i - 6];
    }
    for (int i = 0; i < 64; i++) {
        W1[i] = W[i] ^ W[i + 4];
    }

    uint32_t A = V[0], B_ = V[1], C = V[2], D = V[3], E = V[4], F = V[5], G = V[6], H = V[7];
    for (int j = 0; j < 64; j++) {
        uint32_t SS1 = ROTL((ROTL(A, 12) + E + ROTL(T_j[j], j % 32)) % 0x100000000, 7);
        uint32_t SS2 = SS1 ^ ROTL(A, 12);
        uint32_t TT1 = ((j < 16) ? FF0(A, B_, C) : FF1(A, B_, C)) + D + SS2 + W1[j];
        uint32_t TT2 = ((j < 16) ? GG0(E, F, G) : GG1(E, F, G)) + H + SS1 + W[j];
        D = C; C = ROTL(B_, 9); B_ = A; A = TT1;
        H = G; G = ROTL(F, 19); F = E; E = P0(TT2);
    }

    V[0] ^= A; V[1] ^= B_; V[2] ^= C; V[3] ^= D;
    V[4] ^= E; V[5] ^= F; V[6] ^= G; V[7] ^= H;
}

// ---------------- SM3 哈希函数 ----------------
vector<uint8_t> sm3_hash(const uint8_t* message, size_t len) {
    uint64_t bit_len = len * 8;
    size_t padded_len = ((len + 1 + 8 + 63) / 64) * 64;
    vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), message, len);
    padded[len] = 0x80;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 1 - i] = (bit_len >> (8 * i)) & 0xff;
    }

    uint32_t V[8];
    memcpy(V, IV, sizeof(IV));
    for (size_t i = 0; i < padded.size(); i += 64) {
        sm3_compress(V, padded.data() + i);
    }

    vector<uint8_t> digest(32);
    for (int i = 0; i < 8; i++) {
        digest[4 * i] = (V[i] >> 24) & 0xff;
        digest[4 * i + 1] = (V[i] >> 16) & 0xff;
        digest[4 * i + 2] = (V[i] >> 8) & 0xff;
        digest[4 * i + 3] = V[i] & 0xff;
    }
    return digest;
}

// ---------------- 支持自定义 IV 的哈希 ----------------
vector<uint8_t> sm3_hash_with_iv(const uint8_t* message, size_t len, const uint32_t IV_in[8]) {
    uint64_t bit_len = len * 8;
    size_t padded_len = ((len + 1 + 8 + 63) / 64) * 64;
    vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), message, len);
    padded[len] = 0x80;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 1 - i] = (bit_len >> (8 * i)) & 0xff;
    }

    uint32_t V[8];
    memcpy(V, IV_in, sizeof(V));
    for (size_t i = 0; i < padded.size(); i += 64) {
        sm3_compress(V, padded.data() + i);
    }

    vector<uint8_t> digest(32);
    for (int i = 0; i < 8; i++) {
        digest[4 * i] = (V[i] >> 24) & 0xff;
        digest[4 * i + 1] = (V[i] >> 16) & 0xff;
        digest[4 * i + 2] = (V[i] >> 8) & 0xff;
        digest[4 * i + 3] = V[i] & 0xff;
    }
    return digest;
}

// ---------------- Length-Extension Attack 示例 ----------------
int main() {
    string M = "secret_message";  // 原消息
    string append_M = "malicious"; // 要追加的消息

    // 1. 计算原消息的 SM3 哈希
    auto start = high_resolution_clock::now();
    vector<uint8_t> H1 = sm3_hash((uint8_t*)M.data(), M.size());
    auto end = high_resolution_clock::now();
    cout << "Original message hash: ";
    for (auto b : H1) cout << hex << setw(2) << setfill('0') << (int)b;
    cout << endl;
    cout << "Hash time: " << duration_cast<microseconds>(end - start).count() << " us" << endl;

    // 2. length-extension attack 模拟
    // 假设攻击者只知道 H1 和 len(M)，想要计算 SM3(M||append_M)
    // 构造新的 IV
    uint32_t new_IV[8];
    for (int i = 0; i < 8; i++) {
        new_IV[i] = (H1[4 * i] << 24) | (H1[4 * i + 1] << 16) | (H1[4 * i + 2] << 8) | H1[4 * i + 3];
    }

    vector<uint8_t> forged_hash = sm3_hash_with_iv((uint8_t*)append_M.data(), append_M.size(), new_IV);
    cout << "Forged hash (M||append_M): ";
    for (auto b : forged_hash) cout << hex << setw(2) << setfill('0') << (int)b;
    cout << endl;

    return 0;
}
