/* cgtpkpool.h - pubkey target pool: x-coordinate storage and bloom for direct comparison. */
#ifndef CGTPKPOOL_H
#define CGTPKPOOL_H

#include "cgtdef.h"
#include "cgtmath.h"
#include <vector>
#include <set>

/* A single pubkey target: the full point plus metadata.

   Only x goes to the device - it is what the kernel compares, and it is what
   makes the engine fast. But x alone does not pin the scalar: P and -P share an
   x, so a key k whose x matches may really belong to n-k. y is kept here so the
   host can tell those two apart when confirming a hit. */
struct cgt_pktarget {
  cgt_u256    x;               /* x-coordinate of the public key */
  cgt_u256    y;               /* y, for resolving the P / -P ambiguity */
  bool        compressed;      /* true if specified as compressed */
  std::string original;        /* the pubkey hex as given */
};

class CgtPubkeyPool {
public:
  CgtPubkeyPool();
  ~CgtPubkeyPool();

  /* Add a single pubkey (compressed or uncompressed hex). Returns false on parse failure. */
  bool addPubkey(const std::string &pubHex);
  /* Load pubkeys from a text file, one per line, skip empty/comment lines. */
  bool loadFile(const std::string &path, int &loaded, int &failed);

  /* After loading, finalize the bloom filter + exact table. Call once before searching. */
  void finalize(void);

  /* Query: does this x-coordinate match any target? (bloom + exact check) */
  bool match(const cgt_u256 &x) const;

  /* Full-precision lookup: returns the matching target, or NULL. */
  const cgt_pktarget *lookup(const cgt_u256 &x) const;

  /* Raw filter, for uploading the identical bits to the GPU. */
  const uint8_t *bloomData(void)  const { return bloom.empty() ? NULL : &bloom[0]; }
  size_t         bloomBytes(void) const { return bloom.size(); }
  int            bloomHashCount(void) const { return bfHashes; }

  /* X-coordinate list for device upload: packed as uint64_t[4] per target. */
  const uint64_t *xCoordData(void) const { return xCoords.empty() ? NULL : &xCoords[0]; }
  size_t          xCoordCount(void) const { return targets.size(); }

  /* Stats */
  size_t count(void) const { return targets.size(); }
  size_t bloomBits(void) const { return bfBits; }
  double bloomFPR(void) const;      /* estimated false positive rate */
  bool   hasCompressed(void) const;
  bool   hasUncompressed(void) const;

private:
  std::vector<cgt_pktarget> targets;
  std::set<uint64_t>        exact;     /* first 8 bytes of each x-coordinate */
  std::vector<uint64_t>     xCoords;   /* flattened: 4 limbs per target */

  /* Bloom filter: bit-packed array, multiple hash functions */
  std::vector<uint8_t> bloom;
  size_t bfBits;
  int    bfHashes;

  void bloomAdd(const cgt_u256 &x);
  bool bloomTest(const cgt_u256 &x) const;
  uint64_t xPrefix(const cgt_u256 &x) const;
};

#endif /* CGTPKPOOL_H */
