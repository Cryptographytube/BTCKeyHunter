/* cgtpkpool.cpp - pubkey target pool implementation. */
#include "cgtpkpool.h"
#include "cgtdigest.h"
#include <cstring>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <algorithm>

CgtPubkeyPool::CgtPubkeyPool() : bfBits(0), bfHashes(0) {}
CgtPubkeyPool::~CgtPubkeyPool() {}

bool CgtPubkeyPool::addPubkey(const std::string &pubHex) {
  std::string s = cgtTrim(pubHex);
  if (s.empty()) return false;
  cgt_pt p;
  bool comp = true;
  if (!cgtParsePub(s, p, &comp)) return false;
  cgt_pktarget t;
  t.x = p.x;
  t.y = p.y;
  t.compressed = comp;
  t.original = s;
  targets.push_back(t);
  return true;
}

bool CgtPubkeyPool::loadFile(const std::string &path, int &loaded, int &failed) {
  loaded = 0;
  failed = 0;
  std::ifstream f(path.c_str());
  if (!f.is_open()) return false;
  std::string line;
  while (std::getline(f, line)) {
    std::string s = cgtTrim(line);
    if (s.empty() || s[0] == '#') continue;
    /* Only accept hex-looking lines of 66 or 130 chars (compressed/uncompressed pubkeys) */
    if (s.size() != 66 && s.size() != 130) { failed++; continue; }
    bool ok = addPubkey(s);
    if (ok) loaded++; else failed++;
  }
  return true;
}

uint64_t CgtPubkeyPool::xPrefix(const cgt_u256 &x) const {
  /* First 8 bytes (most significant limb in little-endian storage) */
  return x.w[3];
}

/* Hash the x-coordinate for bloom filter placement. Use the same double-hash
   structure as the address pool (Kirsch-Mitzenmacher), but applied to the
   x-coordinate bytes instead of hash160. */
static void cgtXBloomHashes(const cgt_u256 &x, uint64_t &h1, uint64_t &h2) {
  /* x is stored little-endian: w[0] is low, w[3] is high. Pack into two
     64-bit values the same way the device will see them. */
  h1 = x.w[0] ^ x.w[1];
  h2 = x.w[2] ^ x.w[3];
  h2 ^= h1 * 0x9E3779B97F4A7C15ULL;
  h2 |= 1ULL;
}

void CgtPubkeyPool::bloomAdd(const cgt_u256 &x) {
  uint64_t a, b;
  cgtXBloomHashes(x, a, b);
  for (int i = 0; i < bfHashes; i++) {
    uint64_t bit = (a + (uint64_t)i * b) & (bfBits - 1);
    bloom[bit >> 3] |= (uint8_t)(1u << (bit & 7));
  }
}

bool CgtPubkeyPool::bloomTest(const cgt_u256 &x) const {
  uint64_t a, b;
  cgtXBloomHashes(x, a, b);
  for (int i = 0; i < bfHashes; i++) {
    uint64_t bit = (a + (uint64_t)i * b) & (bfBits - 1);
    if (!(bloom[bit >> 3] & (uint8_t)(1u << (bit & 7)))) return false;
  }
  return true;
}

void CgtPubkeyPool::finalize(void) {
  size_t n = targets.size();
  if (n == 0) n = 1;
  /* Same sizing as address pool: ~43 bits per target for 1e-9 FPR */
  size_t want = n * 48;
  if (want < 4096) want = 4096;
  bfBits = 4096;
  while (bfBits < want) bfBits <<= 1;
  bfHashes = 8;
  bloom.assign((bfBits + 7) / 8, 0);
  exact.clear();
  xCoords.clear();
  xCoords.reserve(targets.size() * 4);
  for (size_t i = 0; i < targets.size(); i++) {
    bloomAdd(targets[i].x);
    exact.insert(xPrefix(targets[i].x));
    /* Flatten x-coordinate into uint64_t[4] for device upload */
    for (int j = 0; j < 4; j++) xCoords.push_back(targets[i].x.w[j]);
  }
}

bool CgtPubkeyPool::match(const cgt_u256 &x) const {
  if (bfBits == 0) return false;
  if (!bloomTest(x)) return false;
  return exact.find(xPrefix(x)) != exact.end();
}

const cgt_pktarget *CgtPubkeyPool::lookup(const cgt_u256 &x) const {
  if (bfBits == 0) return NULL;
  if (!bloomTest(x)) return NULL;
  if (exact.find(xPrefix(x)) == exact.end()) return NULL;
  /* Prefix can collide; confirm all 32 bytes */
  for (size_t i = 0; i < targets.size(); i++) {
    bool eq = true;
    for (int j = 0; j < 4; j++) {
      if (targets[i].x.w[j] != x.w[j]) { eq = false; break; }
    }
    if (eq) return &targets[i];
  }
  return NULL;
}

bool CgtPubkeyPool::hasCompressed(void) const {
  for (size_t i = 0; i < targets.size(); i++)
    if (targets[i].compressed) return true;
  return false;
}

bool CgtPubkeyPool::hasUncompressed(void) const {
  for (size_t i = 0; i < targets.size(); i++)
    if (!targets[i].compressed) return true;
  return false;
}

double CgtPubkeyPool::bloomFPR(void) const {
  if (bfBits == 0 || targets.empty()) return 0.0;
  double k = (double)bfHashes;
  double m = (double)bfBits;
  double n = (double)targets.size();
  double x = 1.0 - exp(-k * n / m);
  return pow(x, k);
}
