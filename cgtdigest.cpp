/* cgtdigest.cpp - SHA-256, RIPEMD-160 and Base58Check. */
#include "cgtdigest.h"
#include <cstring>

/* ================= SHA-256 ================= */

static const uint32_t CGT_SHA_K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static inline uint32_t cgtRotR(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

#define CGT_S0(x)  (cgtRotR(x, 2)  ^ cgtRotR(x, 13) ^ cgtRotR(x, 22))
#define CGT_S1(x)  (cgtRotR(x, 6)  ^ cgtRotR(x, 11) ^ cgtRotR(x, 25))
#define CGT_G0(x)  (cgtRotR(x, 7)  ^ cgtRotR(x, 18) ^ ((x) >> 3))
#define CGT_G1(x)  (cgtRotR(x, 17) ^ cgtRotR(x, 19) ^ ((x) >> 10))
#define CGT_CH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define CGT_MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

static void cgtShaBlock(uint32_t s[8], const uint8_t blk[64]) {
  uint32_t w[64];
  for (int i = 0; i < 16; i++)
    w[i] = ((uint32_t)blk[4*i] << 24) | ((uint32_t)blk[4*i+1] << 16) |
           ((uint32_t)blk[4*i+2] << 8) | (uint32_t)blk[4*i+3];
  for (int i = 16; i < 64; i++)
    w[i] = CGT_G1(w[i-2]) + w[i-7] + CGT_G0(w[i-15]) + w[i-16];

  uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
  for (int i = 0; i < 64; i++) {
    uint32_t t1 = h + CGT_S1(e) + CGT_CH(e,f,g) + CGT_SHA_K[i] + w[i];
    uint32_t t2 = CGT_S0(a) + CGT_MAJ(a,b,c);
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

void cgtSha256Init(cgt_sha256_ctx &ctx) {
  ctx.s[0]=0x6a09e667; ctx.s[1]=0xbb67ae85; ctx.s[2]=0x3c6ef372; ctx.s[3]=0xa54ff53a;
  ctx.s[4]=0x510e527f; ctx.s[5]=0x9b05688c; ctx.s[6]=0x1f83d9ab; ctx.s[7]=0x5be0cd19;
  ctx.bitlen = 0;
  ctx.bufused = 0;
}

void cgtSha256Update(cgt_sha256_ctx &ctx, const uint8_t *data, size_t len) {
  ctx.bitlen += (uint64_t)len * 8;
  while (len) {
    size_t take = 64 - ctx.bufused;
    if (take > len) take = len;
    memcpy(ctx.buf + ctx.bufused, data, take);
    ctx.bufused += take;
    data += take;
    len  -= take;
    if (ctx.bufused == 64) { cgtShaBlock(ctx.s, ctx.buf); ctx.bufused = 0; }
  }
}

void cgtSha256Final(cgt_sha256_ctx &ctx, uint8_t out[32]) {
  uint64_t bits = ctx.bitlen;
  uint8_t pad = 0x80;
  cgtSha256Update(ctx, &pad, 1);
  ctx.bitlen = bits;              /* padding must not count toward length */
  uint8_t zero = 0;
  while (ctx.bufused != 56) { cgtSha256Update(ctx, &zero, 1); ctx.bitlen = bits; }
  uint8_t lb[8];
  for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (56 - 8 * i));
  cgtSha256Update(ctx, lb, 8);
  for (int i = 0; i < 8; i++) {
    out[4*i]   = (uint8_t)(ctx.s[i] >> 24);
    out[4*i+1] = (uint8_t)(ctx.s[i] >> 16);
    out[4*i+2] = (uint8_t)(ctx.s[i] >> 8);
    out[4*i+3] = (uint8_t)(ctx.s[i]);
  }
}

void cgtSha256(const uint8_t *data, size_t len, uint8_t out[32]) {
  cgt_sha256_ctx c;
  cgtSha256Init(c);
  cgtSha256Update(c, data, len);
  cgtSha256Final(c, out);
}

/* ================= RIPEMD-160 ================= */

static inline uint32_t cgtRotL(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

/* message word order, left and right lines */
static const uint8_t CGT_RL[80] = {
  0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
  7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
  3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
  1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
  4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};
static const uint8_t CGT_RR[80] = {
  5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
  6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
  15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
  8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
  12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};
/* rotation amounts */
static const uint8_t CGT_SL[80] = {
  11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
  7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
  11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
  11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
  9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};
static const uint8_t CGT_SR[80] = {
  8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
  9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
  9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
  15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
  8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

static inline uint32_t cgtRmdF(int j, uint32_t x, uint32_t y, uint32_t z) {
  if (j < 16) return x ^ y ^ z;
  if (j < 32) return (x & y) | (~x & z);
  if (j < 48) return (x | ~y) ^ z;
  if (j < 64) return (x & z) | (y & ~z);
  return x ^ (y | ~z);
}

static inline uint32_t cgtRmdKL(int j) {
  if (j < 16) return 0x00000000;
  if (j < 32) return 0x5A827999;
  if (j < 48) return 0x6ED9EBA1;
  if (j < 64) return 0x8F1BBCDC;
  return 0xA953FD4E;
}
static inline uint32_t cgtRmdKR(int j) {
  if (j < 16) return 0x50A28BE6;
  if (j < 32) return 0x5C4DD124;
  if (j < 48) return 0x6D703EF3;
  if (j < 64) return 0x7A6D76E9;
  return 0x00000000;
}

static void cgtRmdBlock(uint32_t s[5], const uint8_t blk[64]) {
  uint32_t x[16];
  for (int i = 0; i < 16; i++)
    x[i] = (uint32_t)blk[4*i] | ((uint32_t)blk[4*i+1] << 8) |
           ((uint32_t)blk[4*i+2] << 16) | ((uint32_t)blk[4*i+3] << 24);

  uint32_t al=s[0],bl=s[1],cl=s[2],dl=s[3],el=s[4];
  uint32_t ar=al, br=bl, cr=cl, dr=dl, er=el;

  for (int j = 0; j < 80; j++) {
    uint32_t t = cgtRotL(al + cgtRmdF(j, bl, cl, dl) + x[CGT_RL[j]] + cgtRmdKL(j), CGT_SL[j]) + el;
    al=el; el=dl; dl=cgtRotL(cl,10); cl=bl; bl=t;
    t = cgtRotL(ar + cgtRmdF(79-j, br, cr, dr) + x[CGT_RR[j]] + cgtRmdKR(j), CGT_SR[j]) + er;
    ar=er; er=dr; dr=cgtRotL(cr,10); cr=br; br=t;
  }
  uint32_t t = s[1] + cl + dr;
  s[1] = s[2] + dl + er;
  s[2] = s[3] + el + ar;
  s[3] = s[4] + al + br;
  s[4] = s[0] + bl + cr;
  s[0] = t;
}

void cgtRipemd160Init(cgt_ripemd160_ctx &ctx) {
  ctx.s[0]=0x67452301; ctx.s[1]=0xEFCDAB89; ctx.s[2]=0x98BADCFE;
  ctx.s[3]=0x10325476; ctx.s[4]=0xC3D2E1F0;
  ctx.bitlen = 0;
  ctx.bufused = 0;
}

void cgtRipemd160Update(cgt_ripemd160_ctx &ctx, const uint8_t *data, size_t len) {
  ctx.bitlen += (uint64_t)len * 8;
  while (len) {
    size_t take = 64 - ctx.bufused;
    if (take > len) take = len;
    memcpy(ctx.buf + ctx.bufused, data, take);
    ctx.bufused += take;
    data += take;
    len  -= take;
    if (ctx.bufused == 64) { cgtRmdBlock(ctx.s, ctx.buf); ctx.bufused = 0; }
  }
}

void cgtRipemd160Final(cgt_ripemd160_ctx &ctx, uint8_t out[20]) {
  uint64_t bits = ctx.bitlen;
  uint8_t pad = 0x80;
  cgtRipemd160Update(ctx, &pad, 1);
  ctx.bitlen = bits;
  uint8_t zero = 0;
  while (ctx.bufused != 56) { cgtRipemd160Update(ctx, &zero, 1); ctx.bitlen = bits; }
  uint8_t lb[8];
  for (int i = 0; i < 8; i++) lb[i] = (uint8_t)(bits >> (8 * i));   /* little endian */
  cgtRipemd160Update(ctx, lb, 8);
  for (int i = 0; i < 5; i++) {
    out[4*i]   = (uint8_t)(ctx.s[i]);
    out[4*i+1] = (uint8_t)(ctx.s[i] >> 8);
    out[4*i+2] = (uint8_t)(ctx.s[i] >> 16);
    out[4*i+3] = (uint8_t)(ctx.s[i] >> 24);
  }
}

void cgtRipemd160(const uint8_t *data, size_t len, uint8_t out[20]) {
  cgt_ripemd160_ctx c;
  cgtRipemd160Init(c);
  cgtRipemd160Update(c, data, len);
  cgtRipemd160Final(c, out);
}

void cgtHash160(const uint8_t *data, size_t len, uint8_t out[20]) {
  uint8_t sh[32];
  cgtSha256(data, len, sh);
  cgtRipemd160(sh, 32, out);
}

/* ================= Base58Check ================= */

static const char *CGT_B58 =
  "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

/* map a base58 char back to its value, -1 if invalid */
static int cgtB58Val(char c) {
  for (int i = 0; i < 58; i++) if (CGT_B58[i] == c) return i;
  return -1;
}

static std::string cgtB58Encode(const uint8_t *p, size_t n) {
  /* count leading zero bytes; each becomes a '1' */
  size_t zeros = 0;
  while (zeros < n && p[zeros] == 0) zeros++;

  /* repeated division of the big-endian byte string by 58 */
  std::vector<uint8_t> buf(p + zeros, p + n);
  std::string out;
  size_t start = 0;
  while (start < buf.size()) {
    int rem = 0;
    for (size_t i = start; i < buf.size(); i++) {
      int cur = (rem << 8) | buf[i];
      buf[i] = (uint8_t)(cur / 58);
      rem = cur % 58;
    }
    out += CGT_B58[rem];
    while (start < buf.size() && buf[start] == 0) start++;
  }
  for (size_t i = 0; i < zeros; i++) out += '1';
  /* digits were produced least significant first */
  std::string r(out.rbegin(), out.rend());
  return r;
}

std::string cgtBase58CheckEncode(uint8_t version, const uint8_t *payload, size_t plen) {
  std::vector<uint8_t> buf;
  buf.reserve(1 + plen + 4);
  buf.push_back(version);
  buf.insert(buf.end(), payload, payload + plen);
  uint8_t h1[32], h2[32];
  cgtSha256(&buf[0], buf.size(), h1);
  cgtSha256(h1, 32, h2);
  buf.insert(buf.end(), h2, h2 + 4);
  return cgtB58Encode(&buf[0], buf.size());
}

bool cgtBase58CheckDecode(const std::string &s, uint8_t &version,
                          std::vector<uint8_t> &payload) {
  if (s.empty()) return false;
  size_t ones = 0;
  while (ones < s.size() && s[ones] == '1') ones++;

  /* accumulate base58 digits into a big-endian byte vector */
  std::vector<uint8_t> acc;
  for (size_t i = ones; i < s.size(); i++) {
    int v = cgtB58Val(s[i]);
    if (v < 0) return false;
    int carry = v;
    for (size_t j = acc.size(); j-- > 0; ) {
      int cur = acc[j] * 58 + carry;
      acc[j] = (uint8_t)(cur & 0xFF);
      carry = cur >> 8;
    }
    while (carry) {
      acc.insert(acc.begin(), (uint8_t)(carry & 0xFF));
      carry >>= 8;
    }
  }
  std::vector<uint8_t> full(ones, 0);
  full.insert(full.end(), acc.begin(), acc.end());
  if (full.size() < 5) return false;

  /* verify the 4-byte double-SHA256 checksum */
  uint8_t h1[32], h2[32];
  cgtSha256(&full[0], full.size() - 4, h1);
  cgtSha256(h1, 32, h2);
  for (int i = 0; i < 4; i++)
    if (h2[i] != full[full.size() - 4 + i]) return false;

  version = full[0];
  payload.assign(full.begin() + 1, full.end() - 4);
  return true;
}

std::string cgtAddrFromHash160(const uint8_t h160[20]) {
  return cgtBase58CheckEncode(0x00, h160, 20);
}
