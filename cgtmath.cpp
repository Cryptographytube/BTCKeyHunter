/* cgtmath.cpp - 256-bit arithmetic and secp256k1 group operations. */
#include "cgtmath.h"
#include "cgtwide.h"
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <cctype>

/* secp256k1 curve prime: 2^256 - 2^32 - 977 */
static const cgt_u256 cgt_p = {
  {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
   0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}
};

/* secp256k1 curve order n */
static const cgt_u256 cgt_n = {
  {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
   0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL}
};

const cgt_u256 &cgtCurveOrder(void) { return cgt_n; }
const cgt_u256 &cgtFieldPrime(void) { return cgt_p; }

/* Generator G in affine coordinates. cgt_pt has a user-provided constructor,
   so it is not an aggregate -- fill the coordinates on first use instead. */
static const cgt_u256 CGT_GX = {
  {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
   0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}
};
static const cgt_u256 CGT_GY = {
  {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
   0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}
};

static cgt_pt cgt_G;
static bool   cgt_G_ready = false;

const cgt_pt &cgtGenerator(void) {
  if (!cgt_G_ready) {
    cgt_G.x   = CGT_GX;
    cgt_G.y   = CGT_GY;
    cgt_G.inf = false;
    cgt_G_ready = true;
  }
  return cgt_G;
}

/* Ladder table: G * 2^i for i in 0..255 */
static cgt_pt cgt_ladder[256];
static bool cgt_ladder_built = false;

void cgtSetZero(cgt_u256 &a) { memset(a.w, 0, sizeof(a.w)); }

void cgtSetU64(cgt_u256 &a, uint64_t v) {
  a.w[0] = v;
  a.w[1] = a.w[2] = a.w[3] = 0;
}

bool cgtIsZero(const cgt_u256 &a) {
  return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0;
}

int cgtCmp(const cgt_u256 &a, const cgt_u256 &b) {
  for (int i = 3; i >= 0; i--) {
    if (a.w[i] < b.w[i]) return -1;
    if (a.w[i] > b.w[i]) return  1;
  }
  return 0;
}

uint64_t cgtAdd(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  unsigned char c = 0;
  c = cgt_addc(a.w[0], b.w[0], c, &r.w[0]);
  c = cgt_addc(a.w[1], b.w[1], c, &r.w[1]);
  c = cgt_addc(a.w[2], b.w[2], c, &r.w[2]);
  c = cgt_addc(a.w[3], b.w[3], c, &r.w[3]);
  return (uint64_t)c;
}

uint64_t cgtSub(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  unsigned char b0 = 0;
  b0 = cgt_subb(a.w[0], b.w[0], b0, &r.w[0]);
  b0 = cgt_subb(a.w[1], b.w[1], b0, &r.w[1]);
  b0 = cgt_subb(a.w[2], b.w[2], b0, &r.w[2]);
  b0 = cgt_subb(a.w[3], b.w[3], b0, &r.w[3]);
  return (uint64_t)b0;
}

uint64_t cgtAddU64(cgt_u256 &r, const cgt_u256 &a, uint64_t v) {
  unsigned char c = 0;
  c = cgt_addc(a.w[0], v, c, &r.w[0]);
  c = cgt_addc(a.w[1], 0, c, &r.w[1]);
  c = cgt_addc(a.w[2], 0, c, &r.w[2]);
  c = cgt_addc(a.w[3], 0, c, &r.w[3]);
  return (uint64_t)c;
}

/* q = a / d, return a % d. Bit-by-bit long division: 256 iterations, called
   once at start-up to size the block walk, so the simple shape wins over a
   word-at-a-time version that would need a 128/64 divide MSVC does not expose
   portably. The running remainder is kept as 65 bits (carry + rem) because a
   divisor above 2^63 would otherwise lose the top bit on the shift. */
uint64_t cgtDivU64(cgt_u256 &q, const cgt_u256 &a, uint64_t d) {
  cgtSetZero(q);
  if (d == 0) return 0;
  uint64_t rem = 0;
  for (int i = 255; i >= 0; i--) {
    uint64_t carry = rem >> 63;
    rem = (rem << 1) | (uint64_t)cgtGetBit(a, i);
    if (carry || rem >= d) {
      rem -= d;
      q.w[i >> 6] |= 1ULL << (i & 63);
    }
  }
  return rem;
}

/* r = a * v, truncated to 256 bits. */
void cgtMulU64(cgt_u256 &r, const cgt_u256 &a, uint64_t v) {
  uint64_t carry = 0;
  for (int i = 0; i < CGT_LIMBS; i++) {
    uint64_t lo, hi;
    CGT_MUL64(lo, hi, a.w[i], v);
    uint64_t s = lo + carry;
    hi += (s < lo) ? 1ULL : 0ULL;
    r.w[i] = s;
    carry = hi;
  }
}

void cgtShiftRight1(cgt_u256 &a) {
  a.w[0] = (a.w[0] >> 1) | (a.w[1] << 63);
  a.w[1] = (a.w[1] >> 1) | (a.w[2] << 63);
  a.w[2] = (a.w[2] >> 1) | (a.w[3] << 63);
  a.w[3] >>= 1;
}

int cgtGetBit(const cgt_u256 &a, int i) {
  if (i < 0 || i >= 256) return 0;
  return (int)((a.w[i >> 6] >> (i & 63)) & 1ULL);
}

int cgtBitLength(const cgt_u256 &a) {
  for (int i = 3; i >= 0; i--) {
    if (a.w[i]) {
      uint64_t v = a.w[i];
      int n = 0;
      while (v) { v >>= 1; n++; }
      return i * 64 + n;
    }
  }
  return 0;
}

double cgtToDouble(const cgt_u256 &a) {
  double r = 0.0;
  for (int i = 3; i >= 0; i--) r = r * 18446744073709551616.0 + (double)a.w[i];
  return r;
}

/* ---------------- text / byte conversion ---------------- */

static int cgtNib(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool cgtParseHex256(const std::string &txt, cgt_u256 &out) {
  std::string s = cgtTrim(txt);
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
  if (s.empty() || s.size() > 64) return false;
  cgtSetZero(out);
  /* walk from the least significant nibble upward */
  int shift = 0;
  for (int i = (int)s.size() - 1; i >= 0; i--) {
    int v = cgtNib(s[i]);
    if (v < 0) return false;
    out.w[shift >> 6] |= ((uint64_t)v) << (shift & 63);
    shift += 4;
  }
  return true;
}

std::string cgtToHex256(const cgt_u256 &a) {
  char buf[80];
  int n = cgtBitLength(a);
  if (n == 0) return "0";
  std::string r;
  bool lead = true;
  for (int i = 3; i >= 0; i--) {
    if (lead && a.w[i] == 0) continue;
    if (lead) { snprintf(buf, sizeof(buf), "%llX", (unsigned long long)a.w[i]); lead = false; }
    else       snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)a.w[i]);
    r += buf;
  }
  return r;
}

std::string cgtToHex256Pad(const cgt_u256 &a) {
  char buf[80];
  snprintf(buf, sizeof(buf), "%016llX%016llX%016llX%016llX",
           (unsigned long long)a.w[3], (unsigned long long)a.w[2],
           (unsigned long long)a.w[1], (unsigned long long)a.w[0]);
  return std::string(buf);
}

void cgtToBytes32(const cgt_u256 &a, uint8_t out[32]) {
  for (int i = 0; i < 4; i++) {
    uint64_t v = a.w[3 - i];
    for (int j = 0; j < 8; j++) out[i * 8 + j] = (uint8_t)(v >> (56 - 8 * j));
  }
}

void cgtFromBytes32(cgt_u256 &a, const uint8_t in[32]) {
  for (int i = 0; i < 4; i++) {
    uint64_t v = 0;
    for (int j = 0; j < 8; j++) v = (v << 8) | in[i * 8 + j];
    a.w[3 - i] = v;
  }
}

/* ---------------- field arithmetic mod p ---------------- */
/* p = 2^256 - C where C = 0x1000003D1 (= 2^32 + 977). Reduction folds the
   high half back in by multiplying it by C, which is cheap and branch-free. */
static const uint64_t CGT_PC = 0x1000003D1ULL;

/* small constants (compound literals are not valid C++, so name them) */
static const cgt_u256 CGT_K2 = {{2, 0, 0, 0}};
static const cgt_u256 CGT_K7 = {{7, 0, 0, 0}};

/* conditional subtract of p when a >= p */
static void cgtFieldTrim(cgt_u256 &a) {
  cgt_u256 t;
  if (cgtSub(t, a, cgt_p) == 0) a = t;   /* no borrow => a >= p */
}

void cgtFieldAdd(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  uint64_t c = cgtAdd(r, a, b);
  if (c) {
    /* overflowed 2^256: add back C, since 2^256 == C (mod p) */
    cgtAddU64(r, r, CGT_PC);
  }
  cgtFieldTrim(r);
}

void cgtFieldSub(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  uint64_t bw = cgtSub(r, a, b);
  if (bw) {
    /* went negative: add p back */
    cgt_u256 t;
    cgtAdd(t, r, cgt_p);
    r = t;
  }
}

void cgtFieldNeg(cgt_u256 &r, const cgt_u256 &a) {
  if (cgtIsZero(a)) { cgtSetZero(r); return; }
  cgtSub(r, cgt_p, a);
}

/* full 256x256 -> 512 bit schoolbook product */
static void cgtMulWide(uint64_t out[8], const cgt_u256 &a, const cgt_u256 &b) {
  for (int i = 0; i < 8; i++) out[i] = 0;
  for (int i = 0; i < 4; i++) {
    uint64_t carry = 0;
    for (int j = 0; j < 4; j++) {
      uint64_t lo, hi;
      CGT_MUL64(lo, hi, a.w[i], b.w[j]);
      /* out[i+j] += lo + carry, propagate into hi */
      unsigned char c1 = 0;
      c1 = cgt_addc(out[i + j], lo, 0, &out[i + j]);
      hi += c1;
      c1 = cgt_addc(out[i + j], carry, 0, &out[i + j]);
      hi += c1;
      carry = hi;
    }
    out[i + 4] += carry;
  }
}

/* Reduce a 512-bit value mod p using 2^256 == C (mod p), applied twice. */
static void cgtFieldReduce(cgt_u256 &r, const uint64_t w[8]) {
  uint64_t lo[4] = { w[0], w[1], w[2], w[3] };
  uint64_t hi[4] = { w[4], w[5], w[6], w[7] };

  /* pass 1: lo += hi * C  (hi*C is at most 33 bits wider, so keep a 5th limb) */
  uint64_t acc[5] = { lo[0], lo[1], lo[2], lo[3], 0 };
  uint64_t carry = 0;
  for (int i = 0; i < 4; i++) {
    uint64_t m_lo, m_hi;
    CGT_MUL64(m_lo, m_hi, hi[i], CGT_PC);
    unsigned char c = 0;
    c = cgt_addc(acc[i], m_lo, 0, &acc[i]);
    m_hi += c;
    c = cgt_addc(acc[i], carry, 0, &acc[i]);
    m_hi += c;
    carry = m_hi;
  }
  acc[4] = carry;

  /* pass 2: fold the remaining top limb back down */
  uint64_t t_lo, t_hi;
  CGT_MUL64(t_lo, t_hi, acc[4], CGT_PC);
  unsigned char c2 = 0;
  c2 = cgt_addc(acc[0], t_lo, 0, &acc[0]);
  c2 = cgt_addc(acc[1], t_hi, c2, &acc[1]);
  c2 = cgt_addc(acc[2], 0,   c2, &acc[2]);
  c2 = cgt_addc(acc[3], 0,   c2, &acc[3]);

  /* a final carry out of 2^256 folds in as one more C */
  if (c2) {
    unsigned char c3 = 0;
    c3 = cgt_addc(acc[0], CGT_PC, 0, &acc[0]);
    c3 = cgt_addc(acc[1], 0, c3, &acc[1]);
    c3 = cgt_addc(acc[2], 0, c3, &acc[2]);
    c3 = cgt_addc(acc[3], 0, c3, &acc[3]);
  }

  r.w[0] = acc[0]; r.w[1] = acc[1]; r.w[2] = acc[2]; r.w[3] = acc[3];
  cgtFieldTrim(r);
  cgtFieldTrim(r);
}

void cgtFieldMul(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  uint64_t w[8];
  cgtMulWide(w, a, b);
  cgtFieldReduce(r, w);
}

void cgtFieldSqr(cgt_u256 &r, const cgt_u256 &a) {
  cgtFieldMul(r, a, a);
}

/* Inversion by Fermat: a^(p-2) mod p, square-and-multiply over p-2. */
void cgtFieldInv(cgt_u256 &r, const cgt_u256 &a) {
  if (cgtIsZero(a)) { cgtSetZero(r); return; }
  cgt_u256 e;
  cgtSub(e, cgt_p, CGT_K2);   /* e = p - 2 */
  cgt_u256 base = a, acc;
  cgtSetU64(acc, 1);
  for (int i = 0; i < 256; i++) {
    if (cgtGetBit(e, i)) cgtFieldMul(acc, acc, base);
    cgtFieldSqr(base, base);
  }
  r = acc;
}

/* ---------------- scalar arithmetic mod n ---------------- */
static void cgtScalarTrim(cgt_u256 &a) {
  cgt_u256 t;
  if (cgtSub(t, a, cgt_n) == 0) a = t;
}

void cgtScalarAdd(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  uint64_t c = cgtAdd(r, a, b);
  if (c) {
    cgt_u256 t;
    cgtSub(t, r, cgt_n);
    r = t;
  }
  cgtScalarTrim(r);
}

void cgtScalarSub(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b) {
  uint64_t bw = cgtSub(r, a, b);
  if (bw) {
    cgt_u256 t;
    cgtAdd(t, r, cgt_n);
    r = t;
  }
}

void cgtScalarNeg(cgt_u256 &r, const cgt_u256 &a) {
  if (cgtIsZero(a)) { cgtSetZero(r); return; }
  cgtSub(r, cgt_n, a);
}

/* ---------------- affine elliptic curve operations ---------------- */
void cgtPtNeg(cgt_pt &r, const cgt_pt &a) {
  r.x = a.x;
  if (a.inf) { r.inf = true; cgtSetZero(r.y); }
  else { r.inf = false; cgtFieldNeg(r.y, a.y); }
}

bool cgtPtOnCurve(const cgt_pt &a) {
  if (a.inf) return true;
  /* y^2 == x^3 + 7 */
  cgt_u256 y2, x2, x3, rhs;
  cgtFieldSqr(y2, a.y);
  cgtFieldSqr(x2, a.x);
  cgtFieldMul(x3, x2, a.x);
  cgtFieldAdd(rhs, x3, CGT_K7);
  return cgtCmp(y2, rhs) == 0;
}

void cgtPtDouble(cgt_pt &r, const cgt_pt &a) {
  if (a.inf || cgtIsZero(a.y)) { r = cgt_pt(); return; }
  /* lam = 3x^2 / 2y */
  cgt_u256 x2, num, den, lam, t, xr, yr;
  cgtFieldSqr(x2, a.x);
  cgtFieldAdd(num, x2, x2);
  cgtFieldAdd(num, num, x2);          /* 3x^2 */
  cgtFieldAdd(den, a.y, a.y);         /* 2y   */
  cgtFieldInv(den, den);
  cgtFieldMul(lam, num, den);
  /* xr = lam^2 - 2x */
  cgtFieldSqr(xr, lam);
  cgtFieldSub(xr, xr, a.x);
  cgtFieldSub(xr, xr, a.x);
  /* yr = lam(x - xr) - y */
  cgtFieldSub(t, a.x, xr);
  cgtFieldMul(yr, lam, t);
  cgtFieldSub(yr, yr, a.y);
  r.x = xr; r.y = yr; r.inf = false;
}

void cgtPtAdd(cgt_pt &r, const cgt_pt &a, const cgt_pt &b) {
  if (a.inf) { r = b; return; }
  if (b.inf) { r = a; return; }
  if (cgtCmp(a.x, b.x) == 0) {
    if (cgtCmp(a.y, b.y) == 0) { cgtPtDouble(r, a); return; }
    r = cgt_pt();   /* P + (-P) = infinity */
    return;
  }
  /* lam = (y2 - y1) / (x2 - x1) */
  cgt_u256 num, den, lam, xr, yr, t;
  cgtFieldSub(num, b.y, a.y);
  cgtFieldSub(den, b.x, a.x);
  cgtFieldInv(den, den);
  cgtFieldMul(lam, num, den);
  cgtFieldSqr(xr, lam);
  cgtFieldSub(xr, xr, a.x);
  cgtFieldSub(xr, xr, b.x);
  cgtFieldSub(t, a.x, xr);
  cgtFieldMul(yr, lam, t);
  cgtFieldSub(yr, yr, a.y);
  r.x = xr; r.y = yr; r.inf = false;
}

void cgtBuildLadder(void) {
  if (cgt_ladder_built) return;
  cgt_ladder[0] = cgtGenerator();
  for (int i = 1; i < 256; i++) cgtPtDouble(cgt_ladder[i], cgt_ladder[i - 1]);
  cgt_ladder_built = true;
}

const cgt_pt *cgtLadder(void) {
  cgtBuildLadder();
  return cgt_ladder;
}

void cgtPtMulG(cgt_pt &r, const cgt_u256 &k) {
  cgtBuildLadder();
  cgt_pt acc;               /* starts at infinity */
  for (int i = 0; i < 256; i++) {
    if (cgtGetBit(k, i)) {
      cgt_pt t;
      cgtPtAdd(t, acc, cgt_ladder[i]);
      acc = t;
    }
  }
  r = acc;
}

/* ---------------- SEC serialization and parsing ---------------- */

void cgtSerializePub(const cgt_pt &p, bool compressed, std::vector<uint8_t> &out) {
  out.clear();
  if (p.inf) return;
  uint8_t xb[32], yb[32];
  cgtToBytes32(p.x, xb);
  cgtToBytes32(p.y, yb);
  if (compressed) {
    out.resize(33);
    out[0] = (yb[31] & 1) ? 0x03 : 0x02;
    memcpy(&out[1], xb, 32);
  } else {
    out.resize(65);
    out[0] = 0x04;
    memcpy(&out[1], xb, 32);
    memcpy(&out[33], yb, 32);
  }
}

bool cgtParsePub(const std::string &hex, cgt_pt &out, bool *wasCompressed) {
  std::vector<uint8_t> buf;
  if (!cgtHexToBytes(hex, buf)) return false;
  if (buf.size() == 33 && (buf[0] == 0x02 || buf[0] == 0x03)) {
    if (wasCompressed) *wasCompressed = true;
    cgtFromBytes32(out.x, &buf[1]);
    /* recover y from x: y^2 = x^3 + 7 */
    cgt_u256 x2, x3, rhs;
    cgtFieldSqr(x2, out.x);
    cgtFieldMul(x3, x2, out.x);
    cgtFieldAdd(rhs, x3, CGT_K7);
    /* y = sqrt(rhs) = rhs^((p+1)/4) since p = 3 (mod 4) */
    cgt_u256 exp;
    cgtAddU64(exp, cgt_p, 1);
    cgtShiftRight1(exp); cgtShiftRight1(exp);
    cgt_u256 base = rhs, acc2;
    cgtSetU64(acc2, 1);
    for (int i = 0; i < 256; i++) {
      if (cgtGetBit(exp, i)) cgtFieldMul(acc2, acc2, base);
      cgtFieldSqr(base, base);
    }
    out.y = acc2;
    if (((out.y.w[0] & 1) != 0) != (buf[0] == 0x03)) cgtFieldNeg(out.y, out.y);
    out.inf = false;
    return cgtPtOnCurve(out);
  }
  if (buf.size() == 65 && buf[0] == 0x04) {
    if (wasCompressed) *wasCompressed = false;
    cgtFromBytes32(out.x, &buf[1]);
    cgtFromBytes32(out.y, &buf[33]);
    out.inf = false;
    return cgtPtOnCurve(out);
  }
  return false;
}

/* ---------------- helper string utilities ---------------- */

std::string cgtTrim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && isspace((unsigned char)s[a])) a++;
  while (b > a && isspace((unsigned char)s[b - 1])) b--;
  return s.substr(a, b - a);
}

std::string cgtUpper(const std::string &s) {
  std::string r = s;
  for (size_t i = 0; i < r.size(); i++) r[i] = (char)toupper((unsigned char)r[i]);
  return r;
}

bool cgtHexToBytes(const std::string &hex, std::vector<uint8_t> &out) {
  std::string s = cgtTrim(hex);
  if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
  if (s.size() & 1) return false;
  out.resize(s.size() / 2);
  for (size_t i = 0; i < out.size(); i++) {
    int hi = cgtNib(s[2 * i]), lo = cgtNib(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return true;
}

std::string cgtBytesToHex(const uint8_t *p, size_t n) {
  static const char hx[] = "0123456789ABCDEF";
  std::string r;
  r.reserve(n * 2);
  for (size_t i = 0; i < n; i++) {
    r += hx[p[i] >> 4];
    r += hx[p[i] & 15];
  }
  return r;
}

std::string cgtScaleCount(double v) {
  static const char *u[] = { "", "K", "M", "G", "T", "P" };
  int i = 0;
  while (v >= 1000.0 && i < 5) { v /= 1000.0; i++; }
  char buf[64];
  snprintf(buf, sizeof(buf), "%.2f %s", v, u[i]);
  return std::string(buf);
}

std::string cgtHumanYears(double y) {
  char buf[64];
  if (y < 1.0 / 365.25) {
    snprintf(buf, sizeof(buf), "%.1f h", y * 365.25 * 24.0);
    return std::string(buf);
  }
  if (y < 1.0) {
    snprintf(buf, sizeof(buf), "%.1f d", y * 365.25);
    return std::string(buf);
  }
  snprintf(buf, sizeof(buf), "%.4gy", y);
  return std::string(buf);
}