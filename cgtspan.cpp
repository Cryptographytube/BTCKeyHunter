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

CgtWalk::CgtWalk() : total(0), key(0), pos(0), half(1), shuffled(false) {}

void CgtWalk::init(uint64_t count, uint64_t k, bool shuf) {
  total = count;
  key = k;
  pos = 0;
  shuffled = shuf;

  /* Feistel domain: the smallest 2*half-bit space that holds `count`. Keeping
     it that tight bounds the cycle walk - at worst the domain is twice the
     count, so the expected number of hops per block index is under two. */
  int bits = 1;
  while (bits < 64 && (count - 1) >> bits) bits++;
  half = (bits + 1) / 2;
  if (half < 1) half = 1;
  if (half > 31) half = 31;   /* 62-bit domain covers any reachable count */
}

bool CgtWalk::setPosition(uint64_t p) {
  if (p > total) return false;
  pos = p;
  return true;
}

/* One pass of the Feistel network over the 2*half-bit domain. */
uint64_t CgtWalk::permute(uint64_t x) const {
  const uint64_t mask = (1ULL << half) - 1ULL;
  uint64_t l = (x >> half) & mask;
  uint64_t r = x & mask;
  for (int i = 0; i < 4; i++) {
    uint64_t f = cgtMix64(r ^ key ^ ((uint64_t)i << 56)) & mask;
    uint64_t nl = r;
    r = l ^ f;
    l = nl;
  }
  return (l << half) | r;
}

uint64_t CgtWalk::next(void) {
  uint64_t i = pos++;
  if (!shuffled) return i;

  /* Cycle walking: iterate the permutation until it lands inside the real
     block count. Because permute() is a bijection on the whole domain, the
     values it skips form cycles that never re-enter [0, total), so this stays
     a bijection on [0, total) - every block still comes up exactly once. */
  uint64_t x = i;
  for (int guard = 0; guard < 512; guard++) {
    x = permute(x);
    if (x < total) return x;
  }
  return i;   /* unreachable in practice; ascending order is still valid */
}

/* ---------------- checkpoint ---------------- */

CgtCheckpoint::CgtCheckpoint() : path("cgtkey_resume.txt") {}
CgtCheckpoint::~CgtCheckpoint() {}

/* Plain key/value text so a stopped run can be inspected, and so a file from
   an older build is rejected on the version line rather than misread. */
#define CGT_RESUME_MAGIC "CGTKEY-RESUME 2"

bool CgtCheckpoint::load(cgt_resume &st) {
  std::ifstream f(path.c_str());
  if (!f.is_open()) return false;

  std::string line;
  if (!std::getline(f, line)) return false;
  if (cgtTrim(line) != CGT_RESUME_MAGIC) return false;

  bool haveStart = false, haveEnd = false;
  st.stride = st.blocks = st.key = st.pos = st.found = 0;
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
    else if (k == "blocks") st.blocks = strtoull(v.c_str(), NULL, 10);
    else if (k == "key")    st.key    = strtoull(v.c_str(), NULL, 16);
    else if (k == "pos")    st.pos    = strtoull(v.c_str(), NULL, 10);
    else if (k == "found")  st.found  = strtoull(v.c_str(), NULL, 10);
    else if (k == "order")  st.shuffled = (v == "random");
  }
  return haveStart && haveEnd && st.stride != 0 && st.blocks != 0;
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
    f << "blocks " << st.blocks << "\n";
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
