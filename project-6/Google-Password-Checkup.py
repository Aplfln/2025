from phe import paillier
import random
import hashlib

# ------------------ P2: Paillier key generation ------------------
public_key, private_key = paillier.generate_paillier_keypair(n_length=512)

# ------------------ P1: 准备数据 ------------------
V = ["password1", "123456", "qwerty"]
k1 = random.randint(2, 1000)
p = 2**521 - 1  # 大素数，用于盲化

def hash_identifier(x):
    return int(hashlib.sha256(x.encode()).hexdigest(), 16)

# P1 对 V 集合盲化
V_blinded = [pow(hash_identifier(v), k1, p) for v in V]
random.shuffle(V_blinded)

# ------------------ P2: 准备数据 ------------------
W = ["123456", "letmein", "password2"]
risk_values = [5, 10, 7]  # 每个密码的风险值
k2 = random.randint(2, 1000)

# P2 对自己的 W 集合盲化
W_blinded = [pow(hash_identifier(w), k2, p) for w in W]
# Paillier 加密风险值
encrypted_risks = [public_key.encrypt(t) for t in risk_values]

# 双盲化 P1 发来的 V_blinded
Z = [pow(vk1, k2, p) for vk1 in V_blinded]
# P2 发送给 P1: (W_blinded, encrypted_risks) 和 Z
W_data = list(zip(W_blinded, encrypted_risks))
random.shuffle(W_data)
random.shuffle(Z)

# ------------------ P1: 识别交集 ------------------
matched_encrypted = []
for w_b, enc_risk in W_data:
    w_double_blind = pow(w_b, k1, p)
    if w_double_blind in Z:
        matched_encrypted.append(enc_risk)

# 累加加密的风险值
encrypted_sum = matched_encrypted[0]
for enc in matched_encrypted[1:]:
    encrypted_sum += enc

# ------------------ P2: 解密得到总风险 ------------------
total_risk = private_key.decrypt(encrypted_sum)
print("交集的总风险值:", total_risk)
