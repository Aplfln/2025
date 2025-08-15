import hashlib, time, random
from typing import Optional, Tuple, List

# ---------- domain params (same as baseline) ----------
p  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF", 16)
a  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", 16)
b  = int("28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93", 16)
Gx = int("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7", 16)
Gy = int("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0", 16)
n  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123", 16)
h  = 1

# ---------- field helper ----------
def inv_mod(x: int, m: int) -> int:
    x %= m
    if x == 0:
        raise ZeroDivisionError("inverse of 0")
    return pow(x, -1, m)

# ---------- EC affine operations ----------
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

# ---------- basic double-and-add (kept for fallback) ----------
def scalar_mul_double_and_add(k: int, P: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    R = None
    Q = P
    while k:
        if k & 1:
            R = point_add(R, Q)
        Q = point_add(Q, Q)
        k >>= 1
    return R

# ---------- w-NAF and precomputation ----------
def wnaf(k: int, w: int = 5) -> List[int]:
    """Return w-NAF digits (least significant first)."""
    if k == 0:
        return [0]
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
    """Compute odd multiples: [1*P, 3*P, 5*P, ..., (2^w -1) * P]"""
    max_odd = (1 << w) - 1
    pre = []
    cur = P  # 1*P
    pre.append(cur)
    twoP = point_add(P, P)
    # build odd multiples incrementally: next_odd = prev_odd + 2P
    odd = 3
    while odd <= max_odd:
        cur = point_add(cur, twoP)
        pre.append(cur)
        odd += 2
    return pre  # length = 2^{w-1}

def scalar_mul_wnaf(k: int, P: Tuple[int,int], w: int = 5, pre: Optional[List[Tuple[int,int]]] = None) -> Optional[Tuple[int,int]]:
    """Scalar multiplication using w-NAF with precomputed odd multiples of P."""
    if k % n == 0:
        return None
    if pre is None:
        pre = precompute_base(P, w)
    digits = wnaf(k, w)
    R = None
    # process from most significant digit down
    for di in reversed(digits):
        R = point_add(R, R)
        if di != 0:
            if di > 0:
                idx = (di - 1) // 2
                R = point_add(R, pre[idx])
            else:
                idx = ((-di) - 1) // 2
                neg = pre[idx]
                negp = (neg[0], (-neg[1]) % p)
                R = point_add(R, negp)
    return R

# ---------- hash wrapper ----------
def hashfunc(msg: bytes) -> bytes:
    return hashlib.sha256(msg).digest()

# ---------- Optimized SM2 key class ----------
class SM2KeyOptimized:
    def __init__(self, d: Optional[int] = None, w: int = 5):
        if d is None:
            self.d = random.randrange(1, n)
        else:
            self.d = d % n
        self.w = w
        # precompute base table for G
        self.G_pre = precompute_base((Gx, Gy), w)
        # public key P = d*G using optimized mul
        self.P = scalar_mul_wnaf(self.d, (Gx, Gy), w, self.G_pre)
        # optional precompute for P for repeated verify -> compute on demand
        self.P_pre = None

    def sign(self, Z_and_M: bytes, k: Optional[int] = None) -> Tuple[int,int,int]:
        e = int.from_bytes(hashfunc(Z_and_M), 'big') % n
        if k is None:
            k = random.randrange(1, n)
        kG = scalar_mul_wnaf(k, (Gx, Gy), self.w, self.G_pre)
        if kG is None:
            raise ValueError("k*G is inf")
        x1 = kG[0] % n
        r = (e + x1) % n
        if r == 0 or r + k == n:
            return self.sign(Z_and_M, None)
        s = (inv_mod(1 + self.d, n) * (k - r * self.d)) % n
        if s == 0:
            return self.sign(Z_and_M, None)
        return (r, s, k)

    def verify(self, Z_and_M: bytes, sig: Tuple[int,int]) -> bool:
        r, s = sig
        if not (1 <= r <= n-1 and 1 <= s <= n-1):
            return False
        e = int.from_bytes(hashfunc(Z_and_M), 'big') % n
        t = (r + s) % n
        if t == 0:
            return False
        # s*G
        sG = scalar_mul_wnaf(s, (Gx, Gy), self.w, self.G_pre)
        # t*P - we can precompute odd multiples of P if verify called many times
        if self.P_pre is None:
            self.P_pre = precompute_base(self.P, self.w)
        tP = scalar_mul_wnaf(t, self.P, self.w, self.P_pre)
        pt = point_add(sG, tP)
        if pt is None:
            return False
        x2 = pt[0] % n
        return r == (e + x2) % n

# ---------- demo & benchmarks ----------
def demo_and_benchmark():
    print("SM2 optimized demo (w-NAF + precomputed G).")
    key = SM2KeyOptimized()
    print("Generated public key P.x =", hex(key.P[0]))
    ZM = b"example-message-for-sm2"
    # single sign timing
    t0 = time.perf_counter()
    r,s,k = key.sign(ZM)
    t1 = time.perf_counter()
    print("Signature (r,s):", r, s)
    print("Sign time: {:.3f} ms".format((t1 - t0) * 1000))
    ok = key.verify(ZM, (r,s))
    print("Verify OK:", ok)

    # bulk benchmark: compare scalar mult and sign to baseline
    N = 50
    ks = [random.randrange(1, n) for _ in range(N)]

    # optimized scalar mult timing
    t0 = time.perf_counter()
    for kk in ks:
        scalar_mul_wnaf(kk, (Gx, Gy), key.w, key.G_pre)
    t1 = time.perf_counter()
    print("Avg scalar mult (optimized): {:.3f} ms".format(((t1 - t0) / N) * 1000))

    # optimized sign timing
    t0 = time.perf_counter()
    for i in range(N):
        key.sign(ZM)
    t1 = time.perf_counter()
    print("Avg sign (optimized): {:.3f} ms".format(((t1 - t0) / N) * 1000))

if __name__ == "__main__":
    demo_and_benchmark()
