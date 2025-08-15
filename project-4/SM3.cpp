// sm3_timing.cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstring>
#include <stdint.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// ---------------------------
// 基本参数和函数
// ---------------------------
#define ROTL(x,n) (((x) << (n)) | ((x) >> (32-(n))))

#define FF0(x,y,z) ((x) ^ (y) ^ (z))
#define FF1(x,y,z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG0(x,y,z) ((x) ^ (y) ^ (z))
#define GG1(x,y,z) (((x) & (y)) | ((~(x)) & (z)))
#define P0(x) ((x) ^ ROTL((x),9) ^ ROTL((x),17))
#define P1(x) ((x) ^ ROTL((x),15) ^ ROTL((x),23))

const uint32_t IV[8] = {
    0x7380166f,0x4914b2b9,0x172442d7,0xda8a0600,
    0xa96f30bc,0x163138aa,0xe38dee4d,0xb0fb0e4e
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

// ---------------------------
// 压缩函数
// ---------------------------
void sm3_compress(uint32_t V[8], const uint8_t B[64]) {
    uint32_t W[68], W1[64];
    for (int i = 0; i < 16; i++)
        W[i] = (B[i * 4] << 24) | (B[i * 4 + 1] << 16) | (B[i * 4 + 2] << 8) | B[i * 4 + 3];
    for (int i = 16; i < 68; i++)
        W[i] = P1(W[i - 16] ^ W[i - 9] ^ ROTL(W[i - 3], 15)) ^ ROTL(W[i - 13], 7) ^ W[i - 6];
    for (int i = 0; i < 64; i++)
        W1[i] = W[i] ^ W[i + 4];

    uint32_t A = V[0], B_ = V[1], C = V[2], D = V[3];
    uint32_t E = V[4], F = V[5], G = V[6], H = V[7];

    for (int j = 0; j < 64; j++) {
        uint32_t SS1 = ROTL((ROTL(A, 12) + E + ROTL(T_j[j], j % 32)) & 0xFFFFFFFF, 7);
        uint32_t SS2 = SS1 ^ ROTL(A, 12);
        uint32_t TT1 = ((j < 16) ? FF0(A, B_, C) : FF1(A, B_, C)) + D + SS2 + W1[j];
        uint32_t TT2 = ((j < 16) ? GG0(E, F, G) : GG1(E, F, G)) + H + SS1 + W[j];
        TT1 &= 0xFFFFFFFF; TT2 &= 0xFFFFFFFF;

        D = C;
        C = ROTL(B_, 9);
        B_ = A;
        A = TT1;
        H = G;
        G = ROTL(F, 19);
        F = E;
        E = P0(TT2);
    }

    V[0] ^= A; V[1] ^= B_; V[2] ^= C; V[3] ^= D;
    V[4] ^= E; V[5] ^= F; V[6] ^= G; V[7] ^= H;
}

// ---------------------------
// SM3 哈希函数
// ---------------------------
vector<uint8_t> sm3_hash(const uint8_t* message, size_t len) {
    uint64_t bit_len = len * 8;
    size_t padded_len = ((len + 1 + 8 + 63) / 64) * 64;
    vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), message, len);
    padded[len] = 0x80;
    for (int i = 0; i < 8; i++)
        padded[padded_len - 1 - i] = (bit_len >> (8 * i)) & 0xFF;

    uint32_t V[8];
    memcpy(V, IV, sizeof(IV));

    for (size_t i = 0; i < padded.size(); i += 64)
        sm3_compress(V, padded.data() + i);

    vector<uint8_t> digest(32);
    for (int i = 0; i < 8; i++) {
        digest[i * 4] = (V[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (V[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (V[i] >> 8) & 0xFF;
        digest[i * 4 + 3] = V[i] & 0xFF;
    }
    return digest;
}

// ---------------------------
// 主函数与时间统计
// ---------------------------
int main() {
    string input = "202200460099";
    cout << "Input message: " << input << endl;

    auto start = high_resolution_clock::now();
    vector<uint8_t> digest = sm3_hash((uint8_t*)input.data(), input.size());
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count() / 1000.0;

    cout << "SM3 Hash: ";
    for (auto b : digest)
        cout << hex << setw(2) << setfill('0') << (int)b;
    cout << endl;
    cout << "Time used: " << duration << " ms" << endl;

    // 测试大消息效率
    vector<uint8_t> large_msg(1024 * 1024, 'a'); // 1MB
    start = high_resolution_clock::now();
    vector<uint8_t> digest_large = sm3_hash(large_msg.data(), large_msg.size());
    end = high_resolution_clock::now();
    duration = duration_cast<microseconds>(end - start).count() / 1000.0;
    cout << "1MB message time: " << duration << " ms" << endl;

    return 0;
}
