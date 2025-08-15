#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <cstdint>

using namespace std;
using namespace chrono;

// ---------------- SM3 实现  ----------------
#define ROTL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define FF0(x,y,z) ((x) ^ (y) ^ (z))
#define FF1(x,y,z) (((x) & (y)) | ((x) & (z)) | ((y) & (z)))
#define GG0(x,y,z) ((x) ^ (y) ^ (z))
#define GG1(x,y,z) (((x) & (y)) | ((~(x)) & (z)))
#define P0(x) ((x) ^ ROTL((x), 9) ^ ROTL((x), 17))
#define P1(x) ((x) ^ ROTL((x), 15) ^ ROTL((x), 23))

const uint32_t IV[8] = {
    0x7380166f, 0x4914b2b9, 0x172442d7, 0xda8a0600,
    0xa96f30bc, 0x163138aa, 0xe38dee4d, 0xb0fb0e4e
};
const uint32_t T_j[64] = {
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,0x79cc4519,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,
    0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a,0x7a879d8a
};

void sm3_compress(uint32_t V[8], const uint8_t B[64]) {
    uint32_t W[68], W1[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = (uint32_t(B[4 * i]) << 24) | (uint32_t(B[4 * i + 1]) << 16) | (uint32_t(B[4 * i + 2]) << 8) | uint32_t(B[4 * i + 3]);
    }
    for (int i = 16; i < 68; ++i) {
        W[i] = P1(W[i - 16] ^ W[i - 9] ^ ROTL(W[i - 3], 15)) ^ ROTL(W[i - 13], 7) ^ W[i - 6];
    }
    for (int i = 0; i < 64; ++i) W1[i] = W[i] ^ W[i + 4];

    uint32_t A = V[0], B_ = V[1], C = V[2], D = V[3];
    uint32_t E = V[4], F = V[5], G = V[6], H = V[7];
    for (int j = 0; j < 64; ++j) {
        uint32_t SS1 = ROTL((ROTL(A, 12) + E + ROTL(T_j[j], j % 32)), 7);
        uint32_t SS2 = SS1 ^ ROTL(A, 12);
        uint32_t TT1 = ((j < 16) ? FF0(A, B_, C) : FF1(A, B_, C)) + D + SS2 + W1[j];
        uint32_t TT2 = ((j < 16) ? GG0(E, F, G) : GG1(E, F, G)) + H + SS1 + W[j];
        D = C; C = ROTL(B_, 9); B_ = A; A = TT1;
        H = G; G = ROTL(F, 19); F = E; E = P0(TT2);
    }
    V[0] ^= A; V[1] ^= B_; V[2] ^= C; V[3] ^= D;
    V[4] ^= E; V[5] ^= F; V[6] ^= G; V[7] ^= H;
}

vector<uint8_t> sm3_hash(const uint8_t* message, size_t len) {
    uint64_t bit_len = uint64_t(len) * 8;
    size_t padded_len = ((len + 1 + 8 + 63) / 64) * 64;
    vector<uint8_t> padded(padded_len, 0);
    if (len) memcpy(padded.data(), message, len);
    padded[len] = 0x80;
    for (int i = 0; i < 8; ++i) padded[padded_len - 1 - i] = (bit_len >> (8 * i)) & 0xff;

    uint32_t V[8];
    memcpy(V, IV, sizeof(IV));
    for (size_t i = 0; i < padded.size(); i += 64) sm3_compress(V, padded.data() + i);

    vector<uint8_t> digest(32);
    for (int i = 0; i < 8; ++i) {
        digest[4 * i] = (V[i] >> 24) & 0xff;
        digest[4 * i + 1] = (V[i] >> 16) & 0xff;
        digest[4 * i + 2] = (V[i] >> 8) & 0xff;
        digest[4 * i + 3] = V[i] & 0xff;
    }
    return digest;
}
vector<uint8_t> sm3_hash_vec(const vector<uint8_t>& v) { return sm3_hash(v.data(), v.size()); }

void print_hash(const string& label, const vector<uint8_t>& hash) {
    cout << label;
    for (uint8_t b : hash) cout << hex << setw(2) << setfill('0') << (int)b;
    cout << dec << endl;
}

// ---------------- Merkle Tree 类 ----------------
class MerkleTree {
public:
    // 构造函数接收叶子原始数据（未哈希），每个元素是字节向量
    MerkleTree(const vector<vector<uint8_t>>& raw_leaves) {
        build(raw_leaves);
    }

    const vector<uint8_t>& root() const { return root_hash_; }

    // 生成 inclusion proof（仅 sibling hashes，按从叶到根的顺序）
    vector<vector<uint8_t>> generateInclusionProof(size_t leaf_index) const {
        if (leaf_index >= leaf_count_) throw out_of_range("leaf index out of range");
        vector<vector<uint8_t>> proof;
        size_t idx = leaf_index;
        for (size_t level = 0; level < levels_.size() - 1; ++level) {
            const auto& level_nodes = levels_[level];
            size_t sibling_idx = (idx % 2 == 0) ? idx + 1 : idx - 1;
            if (sibling_idx < level_nodes.size()) proof.push_back(level_nodes[sibling_idx]);
            else proof.push_back(level_nodes[idx]); // duplicate case
            idx /= 2;
        }
        return proof;
    }

    // 验证 inclusion proof
    static bool verifyInclusion(const vector<uint8_t>& leaf_data, size_t leaf_index,
        const vector<vector<uint8_t>>& proof, const vector<uint8_t>& expected_root) {
        // leaf prefix 0x00
        vector<uint8_t> cur;
        cur.reserve(1 + leaf_data.size());
        cur.push_back(0x00);
        cur.insert(cur.end(), leaf_data.begin(), leaf_data.end());
        cur = sm3_hash_vec(cur);

        size_t idx = leaf_index;
        for (const auto& sibling_hash : proof) {
            vector<uint8_t> comb;
            comb.reserve(1 + cur.size() + sibling_hash.size());
            comb.push_back(0x01);
            if (idx % 2 == 0) { // cur is left
                comb.insert(comb.end(), cur.begin(), cur.end());
                comb.insert(comb.end(), sibling_hash.begin(), sibling_hash.end());
            }
            else {
                comb.insert(comb.end(), sibling_hash.begin(), sibling_hash.end());
                comb.insert(comb.end(), cur.begin(), cur.end());
            }
            cur = sm3_hash_vec(comb);
            idx /= 2;
        }
        return cur == expected_root;
    }

private:
    size_t leaf_count_ = 0;
    vector<vector<vector<uint8_t>>> levels_; // levels_[0] = leaf-level hashes; 每层为 vector<hash>
    vector<uint8_t> root_hash_;

    // 利用原始叶子数据构建树
    void build(const vector<vector<uint8_t>>& raw_leaves) {
        leaf_count_ = raw_leaves.size();
        if (leaf_count_ == 0) {
            // 空树：可定义为空哈希（hash(empty)）
            root_hash_ = sm3_hash(nullptr, 0);
            levels_.push_back({ root_hash_ });
            return;
        }

        levels_.clear();
        // 先计算叶子哈希（带前缀 0x00）
        vector<vector<uint8_t>> level;
        level.reserve(leaf_count_);
        for (const auto& data : raw_leaves) {
            vector<uint8_t> pref; pref.reserve(1 + data.size());
            pref.push_back(0x00);
            pref.insert(pref.end(), data.begin(), data.end());
            level.push_back(sm3_hash_vec(pref));
        }
        levels_.push_back(level);

        // 逐层构建
        while (levels_.back().size() > 1) {
            const auto& cur = levels_.back();
            size_t n = cur.size();
            vector<vector<uint8_t>> next;
            next.reserve((n + 1) / 2);
            // 如果奇数则复制最后一个
            size_t limit = n;
            bool duplicated = false;
            if (n % 2 == 1) { limit = n - 1; duplicated = true; }
            for (size_t i = 0; i < limit; i += 2) {
                vector<uint8_t> comb; comb.reserve(1 + cur[i].size() + cur[i + 1].size());
                comb.push_back(0x01);
                comb.insert(comb.end(), cur[i].begin(), cur[i].end());
                comb.insert(comb.end(), cur[i + 1].begin(), cur[i + 1].end());
                next.push_back(sm3_hash_vec(comb));
            }
            if (duplicated) {
                // 处理最后一对（最后节点与自身）
                vector<uint8_t> comb; comb.reserve(1 + cur.back().size() * 2);
                comb.push_back(0x01);
                comb.insert(comb.end(), cur.back().begin(), cur.back().end());
                comb.insert(comb.end(), cur.back().begin(), cur.back().end());
                next.push_back(sm3_hash_vec(comb));
            }
            levels_.push_back(move(next));
        }
        root_hash_ = levels_.back()[0];
    }
};

// ---------------- 主程序 ----------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    const int LEAF_COUNT = 100000; // 10w 叶子
    cout << "生成 " << LEAF_COUNT << " 个叶子数据 (string -> bytes)..." << endl;

    // 生成叶子原始数据（示例），使用 string 表示
    vector<vector<uint8_t>> leaves;
    leaves.reserve(LEAF_COUNT);
    for (int i = 0; i < LEAF_COUNT; ++i) {
        string s = "leaf-data-" + to_string(i);
        leaves.emplace_back(s.begin(), s.end());
    }

    cout << "对叶子数据排序（以便支持基于邻居的不存在性证明）..." << endl;
    sort(leaves.begin(), leaves.end());

    cout << "开始构建 Merkle 树 ..." << endl;
    auto t0 = high_resolution_clock::now();
    MerkleTree tree(leaves);
    auto t1 = high_resolution_clock::now();
    double ms = duration_cast<milliseconds>(t1 - t0).count();
    cout << "构建完成，用时: " << ms << " ms" << endl;

    auto root = tree.root();
    print_hash("Merkle 根哈希: ", root);

    // 选择一个目标叶子进行 inclusion proof 测试
    string target = "leaf-data-88888";
    vector<uint8_t> target_bytes(target.begin(), target.end());
    auto it = lower_bound(leaves.begin(), leaves.end(), target_bytes);
    if (it == leaves.end() || *it != target_bytes) {
        cerr << "目标叶子在叶子集合中未找到（应存在），检查排序或构造逻辑。" << endl;
    }
    size_t idx = distance(leaves.begin(), it);
    cout << "目标叶子索引: " << idx << endl;

    auto proof = tree.generateInclusionProof(idx);
    cout << "生成的 inclusion proof 长度: " << proof.size() << endl;
    bool ok = MerkleTree::verifyInclusion(target_bytes, idx, proof, root);
    cout << "包含性证明验证: " << (ok ? "通过" : "失败") << endl;

    // 不存在性证明（通过邻居证明）
    string nonexist = "this-leaf-does-not-exist";
    vector<uint8_t> non_bytes(nonexist.begin(), nonexist.end());
    auto it2 = lower_bound(leaves.begin(), leaves.end(), non_bytes);
    size_t neighbor_index = distance(leaves.begin(), it2);
    if (neighbor_index >= leaves.size()) neighbor_index = leaves.size() - 1;
    cout << "不存在性证明：定位位置应为 index " << distance(leaves.begin(), it2) << ", 选择邻居 index " << neighbor_index << endl;
    auto neighbor_proof = tree.generateInclusionProof(neighbor_index);
    bool neighbor_ok = MerkleTree::verifyInclusion(leaves[neighbor_index], neighbor_index, neighbor_proof, root);
    cout << "用于不存在性证明的邻居存在性验证: " << (neighbor_ok ? "通过" : "失败") << endl;
    if (neighbor_ok) {
        cout << "因此可以断言目标项不在树中（位置已被邻居占据）。" << endl;
    }

    return 0;
}
