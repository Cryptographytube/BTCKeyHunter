/* cgtpool.cpp - target pool implementation. */
#include "cgtpool.h"
#include "cgtdigest.h"
#include <cstring>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <algorithm>

CgtPool::CgtPool() : bfBits(0), bfHashes(0) {}
CgtPool::~CgtPool() {}

bool CgtPool::addAddress(const std::string &addr) {
  std::string s = cgtTrim(addr);
  if (s.empty()) return false;
  uint8_t ver;
  std::vector<uint8_t> pl;
  if (!cgtBase58CheckDecode(s, ver, pl)) return false;
  if (pl.size() != 20) return false;
  /* accept mainnet P2PKH (0x00); other versions decode but are not P2PKH */
  if (ver != 0x00) return false;
  cgt_target t;
  memcpy(t.h160, &pl[0], 20);
  t.fromPubkey = false;
  t.compressed = true;
  t.original   = s;
  targets.push_back(t);
  return true;
}

bool CgtPool::addPubkey(const std::string &pubHex) {
  std::string s = cgtTrim(pubHex);
  if (s.empty()) return false;
  cgt_pt p;
  bool comp = true;
  if (!cgtParsePub(s, p, &comp)) return false;
  std::vector<uint8_t> ser;
  cgtSerializePub(p, comp, ser);
  cgt_target t;
  cgtHash160(&ser[0], ser.size(), t.h160);
  t.fromPubkey = true;
  t.compressed = comp;
  t.original   = s;
  targets.push_back(t);
  return true;
}

bool CgtPool::loadFile(const std::string &path, int &loaded, int &failed) {
  loaded = 0;
  failed = 0;
  std::ifstream f(path.c_str());
  if (!f.is_open()) return false;
  std::string line;
  while (std::getline(f, line)) {
    std::string s = cgtTrim(line);
    if (s.empty() || s[0] == '#') continue;
    bool ok;
    /* hex-looking and 66/130 chars => treat as a pubkey, else an address */
    if (s.size() == 66 || s.size() == 130) ok = addPubkey(s);
    else                                   ok = addAddress(s);
    if (ok) loaded++; else failed++;
  }
  return true;
}

uint64_t CgtPool::h160Prefix(const uint8_t h[20]) const {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v = (v << 8) | h[i];
  return v;
}

/* Two independent 64-bit values give us k derived hash positions cheaply
   (Kirsch-Mitzenmacher: h_i = h1 + i*h2).

   The input is a RIPEMD-160 digest, which is already indistinguishable from
   uniform bits, so there is nothing left for a hash function to improve - the
   words can be read out directly. The earlier version ran a 20-byte FNV chain
   over it, twenty dependent multiplies deep, which on the GPU sat in the way
   of every single candidate. The one mix that is still needed folds the fifth
   word in, since h2 would otherwise ignore four of the twenty bytes.

   Must stay bit-identical to devBloomHashes in cgtgpu.cu. */
static void cgtBloomHashes(const uint8_t h[20], uint64_t &h1, uint64_t &h2) {
  uint32_t w[5];
  for (int i = 0; i < 5; i++)
    w[i] = (uint32_t)h[4*i] | ((uint32_t)h[4*i+1] << 8) |
           ((uint32_t)h[4*i+2] << 16) | ((uint32_t)h[4*i+3] << 24);

  h1 = (uint64_t)w[0] | ((uint64_t)w[1] << 32);
  h2 = (uint64_t)w[2] | ((uint64_t)w[3] << 32);
  h2 ^= (uint64_t)w[4] * 0x9E3779B97F4A7C15ULL;
  h2 |= 1ULL;                    /* keep the stride odd */
}

void CgtPool::bloomAdd(const uint8_t h[20]) {
  uint64_t a, b;
  cgtBloomHashes(h, a, b);
  for (int i = 0; i < bfHashes; i++) {
    uint64_t bit = (a + (uint64_t)i * b) & (bfBits - 1);
    bloom[bit >> 3] |= (uint8_t)(1u << (bit & 7));
  }
}

bool CgtPool::bloomTest(const uint8_t h[20]) const {
  uint64_t a, b;
  cgtBloomHashes(h, a, b);
  for (int i = 0; i < bfHashes; i++) {
    uint64_t bit = (a + (uint64_t)i * b) & (bfBits - 1);
    if (!(bloom[bit >> 3] & (uint8_t)(1u << (bit & 7)))) return false;
  }
  return true;
}

void CgtPool::finalize(void) {
  size_t n = targets.size();
  if (n == 0) n = 1;
  /* size for ~1e-9 false positive rate: m/n = -log2(fpr)/ln2 ~= 43 bits each.
     Rounded up to a power of two so the device can index with a mask - a
     64-bit remainder has no hardware instruction and costs more than the
     hashing it follows. */
  size_t want = n * 48;
  if (want < 4096) want = 4096;
  bfBits = 4096;
  while (bfBits < want) bfBits <<= 1;
  bfHashes = 8;
  bloom.assign((bfBits + 7) / 8, 0);
  exact.clear();
  exact.reserve(targets.size() * 2);
  for (size_t i = 0; i < targets.size(); i++) {
    bloomAdd(targets[i].h160);
    exact.insert(h160Prefix(targets[i].h160));
  }
}

bool CgtPool::match(const uint8_t h160[20]) const {
  if (bfBits == 0) return false;
  if (!bloomTest(h160)) return false;
  return exact.find(h160Prefix(h160)) != exact.end();
}

const cgt_target *CgtPool::lookup(const uint8_t h160[20]) const {
  if (bfBits == 0) return NULL;
  if (!bloomTest(h160)) return NULL;
  if (exact.find(h160Prefix(h160)) == exact.end()) return NULL;
  /* The prefix set can collide in principle, so confirm all 20 bytes. */
  for (size_t i = 0; i < targets.size(); i++)
    if (memcmp(targets[i].h160, h160, 20) == 0) return &targets[i];
  return NULL;
}

bool CgtPool::hasPubkeys(void) const {
  for (size_t i = 0; i < targets.size(); i++)
    if (targets[i].fromPubkey) return true;
  return false;
}

double CgtPool::bloomFPR(void) const {
  if (bfBits == 0 || targets.empty()) return 0.0;
  double k = (double)bfHashes;
  double m = (double)bfBits;
  double n = (double)targets.size();
  double x = 1.0 - exp(-k * n / m);
  return pow(x, k);
}
