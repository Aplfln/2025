import hashlib
import random
from typing import Optional, Tuple, List

# -------- SM2 domain parameters (standard) --------
p  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF", 16)
a  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", 16)
b  = int("28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93", 16)
Gx = int("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7", 16)
Gy = int("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0", 16)
n  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123", 16)

# -------- helpers --------
def inv_mod(x: int, m: int) -> int:
    x %= m
    if x == 0:
        raise ZeroDivisionError("inverse of 0")
    return pow(x, -1, m)

def point_add(P: Optional[Tuple[int,int]], Q: Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    if P is None: return Q
    if Q is None: return P
    x1,y1 = P; x2,y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if P != Q:
        lam = ((y2 - y1) * inv_mod(x2 - x1, p)) % p
    else:
        lam = ((3*x1*x1 + a) * inv_mod(2*y1, p)) % p
    x3 = (lam*lam - x1 - x2) % p
    y3 = (lam*(x1 - x3) - y1) % p
    return (x3, y3)

# w-NAF helpers (used for reasonable-speed scalar multiplication)
def wnaf(k: int, w: int = 5) -> List[int]:
    if k == 0: return [0]
    digits = []
    while k > 0:
        if k & 1:
            mod = k % (1 << w)
            if mod & (1 << (w-1)):
                z = mod - (1 << w)
            else:
                z = mod
            digits.append(z)
            k = k - z
        else:
            digits.append(0)
        k >>= 1
    return digits

def precompute_base(P: Tuple[int,int], w: int = 5) -> List[Tuple[int,int]]:
    max_odd = (1 << w) - 1
    pre = []
    cur = P
    pre.append(cur)
    twoP = point_add(P, P)
    odd = 3
    while odd <= max_odd:
        cur = point_add(cur, twoP)
        pre.append(cur)
        odd += 2
    return pre

def scalar_mul_wnaf(k: int, P: Tuple[int,int], w: int = 5, pre: Optional[List[Tuple[int,int]]] = None) -> Optional[Tuple[int,int]]:
    if k % n == 0: return None
    if pre is None: pre = precompute_base(P, w)
    digits = wnaf(k, w)
    R = None
    for di in reversed(digits):
        R = point_add(R, R)
        if di != 0:
            if di > 0:
                idx = (di - 1) // 2
                R = point_add(R, pre[idx])
            else:
                idx = ((-di) - 1) // 2
                neg = pre[idx]
                R = point_add(R, (neg[0], (-neg[1]) % p))
    return R

# -------- hash wrapper: double SHA-256 (Bitcoin style) --------
def double_sha256(msg: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(msg).digest()).digest()

# -------- SM2 key & sign/verify (simple) --------
class SM2Key:
    def __init__(self, d: Optional[int] = None, w: int = 5):
        if d is None:
            self.d = random.randrange(1, n)
        else:
            self.d = d % n
        self.w = w
        self.G_pre = precompute_base((Gx, Gy), self.w)
        self.P = scalar_mul_wnaf(self.d, (Gx, Gy), self.w, self.G_pre)

    def get_private_hex(self) -> str:
        return hex(self.d)

    def get_compressed_pub_hex(self) -> str:
        # compressed: 02 + X if y even, 03 + X if y odd
        if self.P is None:
            return ""
        x, y = self.P
        prefix = b'\x02' if (y % 2 == 0) else b'\x03'
        x_bytes = x.to_bytes(32, 'big')
        return (prefix + x_bytes).hex()

    def sign_prehashed(self, z_bytes: bytes, k: Optional[int] = None) -> Tuple[int,int,int]:
        # z_bytes is already the digest (pre-hashed)
        e = int.from_bytes(z_bytes, 'big') % n
        if k is None:
            k = random.randrange(1, n)
        kG = scalar_mul_wnaf(k, (Gx, Gy), self.w, self.G_pre)
        if kG is None:
            raise ValueError("k*G at infinity")
        x1 = kG[0] % n
        r = (e + x1) % n
        if r == 0 or r + k == n:
            return self.sign_prehashed(z_bytes, None)
        s = (inv_mod(1 + self.d, n) * (k - r * self.d)) % n
        if s == 0:
            return self.sign_prehashed(z_bytes, None)
        return (r, s, k)

    def verify_prehashed(self, z_bytes: bytes, sig: Tuple[int,int]) -> bool:
        r, s = sig
        if not (1 <= r <= n-1 and 1 <= s <= n-1):
            return False
        e = int.from_bytes(z_bytes, 'big') % n
        t = (r + s) % n
        if t == 0:
            return False
        sG = scalar_mul_wnaf(s, (Gx, Gy), self.w, self.G_pre)
        tP = scalar_mul_wnaf(t, self.P, self.w)
        pt = point_add(sG, tP)
        if pt is None:
            return False
        x2 = pt[0] % n
        return r == (e + x2) % n

# we need inv_mod again
def inv_mod(x: int, m: int) -> int:
    x %= m
    if x == 0:
        raise ZeroDivisionError("inverse of 0")
    return pow(x, -1, m)

# -------- main demo (mirrors your sig.py flow) --------
if __name__ == "__main__":
    print("=== mimic Satoshi-style double-SHA256 but sign with SM2 (local experiment) ===")
    # generate key
    priv = SM2Key()
    priv_hex = priv.get_private_hex()
    pub_hex = priv.get_compressed_pub_hex()
    print("Private key:", priv_hex)
    print("Compressed public key:", pub_hex)

    # message
    message_string = "The Times 03/Jan/2009 Chancellor on brink of second bailout for banks"
    message_bytes = message_string.encode('utf-8')
    z = double_sha256(message_bytes)
    print("Original message:", message_string)
    print("Message double-SHA256:", z.hex())

    # sign (prehashed)
    print("=== sign (prehashed) ===")
    r, s, k = priv.sign_prehashed(z)
    print("Signature r:", r)
    print("Signature s:", s)
    print("Signature (r|s) hex:", (r.to_bytes(32,'big') + s.to_bytes(32,'big')).hex())

    # verify
    ok = priv.verify_prehashed(z, (r, s))
    print("Verify OK:", ok)

    # tamper test
    print("=== tamper test ===")
    tampered_message_bytes = b"This message has been tampered with!"
    z2 = double_sha256(tampered_message_bytes)
    print("Tampered message double-SHA256:", z2.hex())
    ok2 = priv.verify_prehashed(z2, (r, s))
    print("Verify tampered (should be False):", ok2)
