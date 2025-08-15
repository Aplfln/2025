import hashlib, random
from typing import Optional, Tuple, List

# ---------- SM2 domain params ----------
p  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF", 16)
a  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", 16)
b  = int("28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93", 16)
Gx = int("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7", 16)
Gy = int("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0", 16)
n  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123", 16)

# ---------- modular inverse ----------
def inv_mod(x: int, m: int) -> int:
    x %= m
    if x == 0:
        raise ZeroDivisionError("inverse of 0")
    return pow(x, -1, m)

# ---------- EC ops (affine) ----------
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

# ---------- w-NAF helpers ----------
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

# ---------- hash wrapper ----------
def hashfunc(msg: bytes) -> bytes:
    return hashlib.sha256(msg).digest()

# ---------- SM2 key class (optimized) ----------
class SM2Key:
    def __init__(self, d: Optional[int] = None, w: int = 5):
        if d is None:
            self.d = random.randrange(1, n)
        else:
            self.d = d % n
        self.w = w
        self.G_pre = precompute_base((Gx, Gy), w)
        self.P = scalar_mul_wnaf(self.d, (Gx, Gy), w, self.G_pre)

    def sign_with_nonce(self, ZM: bytes, k: int) -> Tuple[int,int]:
        e = int.from_bytes(hashfunc(ZM), 'big') % n
        kG = scalar_mul_wnaf(k, (Gx, Gy), self.w, self.G_pre)
        if kG is None:
            raise ValueError("k*G at infinity")
        x1 = kG[0] % n
        r = (e + x1) % n
        s = (inv_mod(1 + self.d, n) * (k - r * self.d)) % n
        return (r, s)

    def sign(self, ZM: bytes) -> Tuple[int,int,int]:
        # normal sign returns r,s,k
        k = random.randrange(1, n)
        r, s = self.sign_with_nonce(ZM, k)
        return r, s, k

    def verify(self, ZM: bytes, sig: Tuple[int,int]) -> bool:
        r, s = sig
        if not (1 <= r <= n-1 and 1 <= s <= n-1): return False
        e = int.from_bytes(hashfunc(ZM), 'big') % n
        t = (r + s) % n
        if t == 0: return False
        sG = scalar_mul_wnaf(s, (Gx, Gy), self.w, self.G_pre)
        tP = scalar_mul_wnaf(t, self.P, self.w)
        pt = point_add(sG, tP)
        if pt is None: return False
        x2 = pt[0] % n
        return r == (e + x2) % n

# ---------- recovery function ----------
def recover_privkey_from_reused_k(r1: int, s1: int, r2: int, s2: int) -> Optional[int]:
    num = (s1 - s2) % n
    den = (s2 + r2 - s1 - r1) % n
    if den % n == 0:
        return None
    return (num * inv_mod(den, n)) % n

# ---------- demo PoC ----------
def demo():
    print("=== SM2 nonce reuse PoC demo ===")
    # generate victim key
    victim = SM2Key()
    print("Victim public key P.x =", hex(victim.P[0]))

    # two messages
    M1 = b"Message for nonce reuse test #1"
    M2 = b"Message for nonce reuse test #2"

    # attacker forces or observes same nonce k used twice (simulated here)
    k_bad = random.randrange(1, n)
    r1, s1 = victim.sign_with_nonce(M1, k_bad)
    r2, s2 = victim.sign_with_nonce(M2, k_bad)
    print("Signature1 r,s:", r1, s1)
    print("Signature2 r,s:", r2, s2)

    # attacker computes
    recovered_d = recover_privkey_from_reused_k(r1, s1, r2, s2)
    if recovered_d is None:
        print("Recovery failed (degenerate case where denominator == 0).")
        return
    print("Recovered d:", hex(recovered_d))
    print("Actual victim d:", hex(victim.d))
    print("Match?:", recovered_d == victim.d)

    # use recovered key to forge signature on new message
    attacker = SM2Key(d=recovered_d)
    forged_r, forged_s, = attacker.sign_with_nonce(b"Attacker forged message", random.randrange(1, n))
    print("Forged signature (r,s):", forged_r, forged_s)
    ok = victim.verify(b"Attacker forged message", (forged_r, forged_s))
    print("Verify of forged signature (should be True):", ok)

if __name__ == "__main__":
    demo()
