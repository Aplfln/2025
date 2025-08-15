import hashlib, time, random
from typing import Optional, Tuple

# ---------- SM2 domain parameters (standard) ----------
p  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF", 16)
a  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFC", 16)
b  = int("28E9FA9E9D9F5E344D5A9E4BCF6509A7F39789F515AB8F92DDBCBD414D940E93", 16)
Gx = int("32C4AE2C1F1981195F9904466A39C9948FE30BBFF2660BE1715A4589334C74C7", 16)
Gy = int("BC3736A2F4F6779C59BDCEE36B692153D0A9877CC62A474002DF32E52139F0A0", 16)
n  = int("FFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFF7203DF6B21C6052B53BBF40939D54123", 16)
h  = 1

# ---------- modular inverse ----------
def inv_mod(x: int, m: int) -> int:
    # Python 3.8+: pow(x, -1, m) works; fallback to extended gcd if needed
    x %= m
    if x == 0:
        raise ZeroDivisionError("inverse of 0")
    try:
        return pow(x, -1, m)
    except TypeError:
        # fallback extended Euclid
        def egcd(a,b):
            if b==0: return (1,0,a)
            x,y,g=egcd(b, a%b)
            return (y, x-(a//b)*y, g)
        inv,_,g = egcd(x, m)
        if g != 1:
            raise ValueError("No modular inverse")
        return inv % m

# ---------- curve tests ----------
def is_on_curve(P: Optional[Tuple[int,int]]) -> bool:
    if P is None: return True
    x,y = P
    return (y*y - (x*x*x + a*x + b)) % p == 0

# ---------- point operations (affine) ----------
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

def scalar_mul_double_and_add(k: int, P: Tuple[int,int]) -> Optional[Tuple[int,int]]:
    # simple double-and-add; bit-scan from LSB to MSB
    R = None
    Q = P
    while k:
        if k & 1:
            R = point_add(R, Q)
        Q = point_add(Q, Q)
        k >>= 1
    return R

# ---------- hash wrapper (replace with SM3 if available) ----------
def hashfunc(msg: bytes) -> bytes:
    # demo using SHA-256 - replace with your SM3 function returning bytes
    return hashlib.sha256(msg).digest()

# ---------- SM2 key class (baseline) ----------
class SM2KeyBaseline:
    def __init__(self, d: Optional[int] = None):
        if d is None:
            self.d = random.randrange(1, n)
        else:
            self.d = d % n
        self.P = scalar_mul_double_and_add(self.d, (Gx, Gy))
        assert self.P is not None and is_on_curve(self.P)

    def sign(self, Z_and_M: bytes, k: Optional[int] = None) -> Tuple[int,int,int]:
        e = int.from_bytes(hashfunc(Z_and_M), 'big') % n
        if k is None:
            k = random.randrange(1, n)
        kG = scalar_mul_double_and_add(k, (Gx, Gy))
        if kG is None:
            raise ValueError("k*G at infinity")
        x1 = kG[0] % n
        r = (e + x1) % n
        if r == 0 or r + k == n:
            return self.sign(Z_and_M, None)
        inv1pd = inv_mod(1 + self.d, n)
        s = (inv1pd * (k - r * self.d)) % n
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
        sG = scalar_mul_double_and_add(s, (Gx, Gy))
        tP = scalar_mul_double_and_add(t, self.P)
        pt = point_add(sG, tP)
        if pt is None:
            return False
        x2 = pt[0] % n
        return r == (e + x2) % n

# ---------- timing & demo ----------
def demo():
    print("SM2 baseline demo (double-and-add scalar mult).")
    # create key
    key = SM2KeyBaseline()
    print("Generated public key P.x =", hex(key.P[0]))

    # sample messages
    ZM = b"example-message-for-sm2"
    # single signature timing
    t0 = time.perf_counter()
    r,s,k = key.sign(ZM)
    t1 = time.perf_counter()
    print("Signature (r,s):", r, s)
    print("Sign time: {:.3f} ms".format((t1-t0)*1000))

    # verify
    ok = key.verify(ZM, (r,s))
    print("Verify OK:", ok)

    # bulk timing for scalar multiplication and signing
    N = 50
    # scalar mult timing
    ks = [random.randrange(1,n) for _ in range(N)]
    t0 = time.perf_counter()
    for kk in ks:
        scalar_mul_double_and_add(kk, (Gx, Gy))
    t1 = time.perf_counter()
    print("Avg scalar mult (double-and-add): {:.3f} ms".format(((t1-t0)/N)*1000))

    # sign timing
    t0 = time.perf_counter()
    for i in range(N):
        key.sign(ZM)
    t1 = time.perf_counter()
    print("Avg sign (baseline): {:.3f} ms".format(((t1-t0)/N)*1000))

if __name__ == "__main__":
    demo()
