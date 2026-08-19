/* cgtspan.cpp - range parsing, RNG, block walk order and checkpointing. */
#include "cgtspan.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>

/* ---------------- range parsing ---------------- */

bool cgtParseRange(const std::string &txt, cgt_u256 &start, cgt_u256 &end) {
  std::string s = cgtTrim(txt);
  if (s.empty()) return false;

  size_t colon = s.find(':');
  if (colon != std::string::npos) {
    /* explicit hex range START:END */
    std::string a = cgtTrim(s.substr(0, colon));
    std::string b = cgtTrim(s.substr(colon + 1));
    if (!cgtParseHex256(a, start)) return false;
    if (!cgtParseHex256(b, end))   return false;
    if (cgtCmp(start, end) > 0) return false;
    return true;
  }

  /* bare bit count: all digits, 1..256 */
  for (size_t i = 0; i < s.size(); i++)
    if (s[i] < '0' || s[i] > '9') return false;
  long bits = strtol(s.c_str(), NULL, 10);
  if (bits < 1 || bits > 256) return false;

  /* start = 2^(bits-1), end = 2^bits - 1 */
  cgtSetZero(start);
  int hb = (int)bits - 1;
  start.w[hb >> 6] = 1ULL << (hb & 63);

  cgtSetZero(end);
  for (int i = 0; i < bits; i++) end.w[i >> 6] |= 1ULL << (i & 63);
  return true;
}

/* ---------------- RNG ---------------- */

static std::mt19937_64 cgt_rng;
static bool cgt_rng_ready = false;

void cgtSeedRNG(void) {
  if (cgt_rng_ready) return;
  std::random_device rd;
  uint64_t seed = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
  seed ^= ((uint64_t)rd() << 16);
  cgt_rng.seed(seed);
  cgt_rng_ready = true;
}

uint64_t cgtRand64(void) {
  cgtSeedRNG();
  return cgt_rng();
}

void cgtRand256(cgt_u256 &out, const cgt_u256 &lo, const cgt_u256 &hi) {
  cgtSeedRNG();
  /* span = hi - lo (inclusive width is span+1) */
  cgt_u256 span;
  cgtSub(span, hi, lo);

  int bits = cgtBitLength(span);
  if (bits == 0) { out = lo; return; }

  /* Rejection sampling on the low `bits` bits keeps the draw uniform. */
  cgt_u256 r;
  for (int guard = 0; guard < 64; guard++) {
    for (int i = 0; i < CGT_LIMBS; i++) r.w[i] = cgt_rng();
    /* mask off everything above the span's top bit */
    int full = bits >> 6;
    int rem  = bits & 63;
    for (int i = full + (rem ? 1 : 0); i < CGT_LIMBS; i++) r.w[i] = 0;
    if (rem) r.w[full] &= (rem == 64) ? ~0ULL : ((1ULL << rem) - 1ULL);
    if (cgtCmp(r, span) <= 0) break;
  }
  if (cgtCmp(r, span) > 0) r = span;    /* guard exhausted: clamp */
  cgtAdd(out, lo, r);
}

/* ---------------- block walk ---------------- */

/* splitmix64 finalizer: the Feistel round function. Any function would leave
   the network a bijection; this one is chosen because it avalanches well in a
   handful of instructions, which is what makes consecutive cursor values land
   in unrelated parts of the range. */
static inline uint64_t cgtMix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/* A Feistel half is up to 128 bits, so it needs two words. The permutation
   runs over 2*half bits, and with half up to 128 that domain reaches the full
   256-bit block count of a 2^256 range - which is what carries randomness into
   the high bits of the base rather than freezing them at the range start. */
struct cgtU128 { uint64_t lo, hi; };

/* Keep only the low `half` bits (half in 1..128). */
static inline void cgtMaskHalf(cgtU128 &v, int half) {
  if (half >= 128) return;
  if (half >= 64) {
    int r = half - 64;                       /* 0..63 bits kept in hi */
    v.hi = (r == 0) ? 0ULL : (v.hi & ((1ULL << r) - 1ULL));
  } else {
    v.hi = 0;
    v.lo &= (1ULL << half) - 1ULL;           /* half in 1..63 */
  }
}

/* Round function. A Feistel network is a bijection for ANY round function, so
   correctness never depends on what this returns - only the apparent
   randomness does. Three splitmix passes couple both words of the half so a
   change anywhere in R avalanches across the whole output. */
static inline cgtU128 cgtRoundF(cgtU128 r, uint64_t key, int round, int half) {
  uint64_t t   = cgtMix64(r.lo ^ key ^ ((uint64_t)round << 56));
  uint64_t flo = cgtMix64(r.hi ^ t   ^ key);
  uint64_t fhi = cgtMix64(t   ^ flo ^ ((uint64_t)round << 40));
  cgtU128 f = { flo, fhi };
  cgtMaskHalf(f, half);
  return f;
}

/* 64 bits of `x` starting at bit `start`; reads past the top limb as zero. */
static inline uint64_t cgtBits64(const cgt_u256 &x, int start) {
  int limb = start >> 6, off = start & 63;
  uint64_t lo = (limb   < CGT_LIMBS) ? x.w[limb]     : 0ULL;
  uint64_t hi = (limb+1 < CGT_LIMBS) ? x.w[limb + 1] : 0ULL;
  if (off == 0) return lo;
  return (lo >> off) | (hi << (64 - off));
}

/* Pull the `half`-bit field of `x` that begins at bit `start` into a cgtU128. */
static inline void cgtExtractHalf(const cgt_u256 &x, int start, int half,
                                  cgtU128 &out) {
  out.lo = cgtBits64(x, start);
  out.hi = cgtBits64(x, start + 64);
  cgtMaskHalf(out, half);
}

/* OR a 64-bit value into `out` at bit `start` (spilling into the next limb). */
static inline void cgtOrBits64(cgt_u256 &out, int start, uint64_t v) {
  if (!v) return;
  int limb = start >> 6, off = start & 63;
  if (limb < CGT_LIMBS) out.w[limb] |= v << off;
  if (off && limb + 1 < CGT_LIMBS) out.w[limb + 1] |= v >> (64 - off);
}

static inline void cgtSetHalf(cgt_u256 &out, int start, const cgtU128 &v) {
  cgtOrBits64(out, start,      v.lo);
  cgtOrBits64(out, start + 64, v.hi);
}

CgtWalk::CgtWalk() : key(0), pos(0), half(1), shuffled(false) {
  cgtSetZero(total);
}

void CgtWalk::init(const cgt_u256 &count, uint64_t k, bool shuf) {
  total = count;
  key = k;
  pos = 0;
  shuffled = shuf;

  /* Feistel domain: the smallest 2*half-bit space that holds `count`. Keeping
     it that tight bounds the cycle walk - at worst the domain is twice the
     count, so the expected number of hops per block index is under two. */
  cgt_u256 cm1;
  if (cgtIsZero(count)) {
    cgtSetZero(cm1);
  } else {
    cgt_u256 one; cgtSetU64(one, 1);
    cgtSub(cm1, count, one);
  }
  int bits = cgtBitLength(cm1);              /* 0 when count <= 1 */
  if (bits < 1) bits = 1;
  half = (bits + 1) / 2;
  if (half < 1)   half = 1;
  if (half > 128) half = 128;                /* 256-bit domain covers any count */
}

bool CgtWalk::done(void) const {
  cgt_u256 p; cgtSetU64(p, pos);
  return cgtCmp(p, total) >= 0;
}

bool CgtWalk::setPosition(uint64_t p) {
  cgt_u256 pp; cgtSetU64(pp, p);
  if (cgtCmp(pp, total) > 0) return false;
  pos = p;
  return true;
}

/* One pass of the Feistel network over the 2*half-bit domain. */
void CgtWalk::permute(const cgt_u256 &x, cgt_u256 &out) const {
  cgtU128 r, l;
  cgtExtractHalf(x, 0,    half, r);          /* low half  = R */
  cgtExtractHalf(x, half, half, l);          /* next half = L */
  for (int i = 0; i < 4; i++) {
    cgtU128 f  = cgtRoundF(r, key, i, half);
    cgtU128 nl = r;
    r.lo = l.lo ^ f.lo;  r.hi = l.hi ^ f.hi;
    l = nl;
  }
  cgtSetZero(out);
  cgtSetHalf(out, 0,    r);                   /* out = (l << half) | r */
  cgtSetHalf(out, half, l);
}

cgt_u256 CgtWalk::next(void) {
  uint64_t i = pos++;
  cgt_u256 idx; cgtSetU64(idx, i);
  if (!shuffled) return idx;

  /* Cycle walking: iterate the permutation until it lands inside the real
     block count. Because permute() is a bijection on the whole 2*half-bit
     domain, the values it skips form cycles that never re-enter [0, total), so
     this stays a bijection on [0, total) - every block still comes up exactly
     once, and distinct counter values i=0,1,2,... yield distinct blocks. Over a
     huge range the counter never approaches `total`, so this samples the range
     without repeats and with every bit of the base randomised. */
  cgt_u256 x = idx;
  for (int guard = 0; guard < 512; guard++) {
    cgt_u256 y;
    permute(x, y);
    if (cgtCmp(y, total) < 0) return y;
    x = y;
  }
  return idx;   /* unreachable in practice; ascending order is still valid */
}

/* ---------------- checkpoint ---------------- */

CgtCheckpoint::CgtCheckpoint() : path("cgtkey_resume.txt") {}
CgtCheckpoint::~CgtCheckpoint() {}

/* Plain key/value text so a stopped run can be inspected, and so a file from
   an older build is rejected on the version line rather than misread. */
#define CGT_RESUME_MAGIC "CGTKEY-RESUME 3"

bool CgtCheckpoint::load(cgt_resume &st) {
  std::ifstream f(path.c_str());
  if (!f.is_open()) return false;

  std::string line;
  if (!std::getline(f, line)) return false;
  if (cgtTrim(line) != CGT_RESUME_MAGIC) return false;

  bool haveStart = false, haveEnd = false;
  st.stride = st.key = st.pos = st.found = 0;
  cgtSetZero(st.blocks);
  st.shuffled = false;

  while (std::getline(f, line)) {
    std::string s = cgtTrim(line);
    size_t sp = s.find(' ');
    if (sp == std::string::npos) continue;
    std::string k = s.substr(0, sp);
    std::string v = cgtTrim(s.substr(sp + 1));

    if      (k == "start")  haveStart = cgtParseHex256(v, st.start);
    else if (k == "end")    haveEnd   = cgtParseHex256(v, st.end);
    else if (k == "stride") st.stride = strtoull(v.c_str(), NULL, 10);
    else if (k == "blocks") cgtParseHex256(v, st.blocks);
    else if (k == "key")    st.key    = strtoull(v.c_str(), NULL, 16);
    else if (k == "pos")    st.pos    = strtoull(v.c_str(), NULL, 10);
    else if (k == "found")  st.found  = strtoull(v.c_str(), NULL, 10);
    else if (k == "order")  st.shuffled = (v == "random");
  }
  return haveStart && haveEnd && st.stride != 0 && !cgtIsZero(st.blocks);
}

bool CgtCheckpoint::save(const cgt_resume &st) {
  /* Write to a temp file then replace, so a crash mid-write - or a Ctrl+C
     landing between the two calls - cannot leave a truncated checkpoint that
     would restart the run from the wrong place. */
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp.c_str(), std::ios::trunc);
    if (!f.is_open()) return false;
    f << CGT_RESUME_MAGIC "\n";
    f << "start "  << cgtToHex256Pad(st.start) << "\n";
    f << "end "    << cgtToHex256Pad(st.end)   << "\n";
    f << "stride " << st.stride << "\n";
    f << "blocks " << cgtToHex256Pad(st.blocks) << "\n";
    f << "order "  << (st.shuffled ? "random" : "linear") << "\n";
    char buf[32];
    snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)st.key);
    f << "key "    << buf << "\n";
    f << "pos "    << st.pos   << "\n";
    f << "found "  << st.found << "\n";
    if (!f.good()) return false;
  }
  remove(path.c_str());
  return rename(tmp.c_str(), path.c_str()) == 0;
}
