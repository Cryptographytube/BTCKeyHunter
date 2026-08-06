/* cgtpool.h - target pool: load addresses/pubkeys, build bloom + exact tables. */
#ifndef CGTPOOL_H
#define CGTPOOL_H

#include "cgtdef.h"
#include "cgtmath.h"
#include <vector>
#include <unordered_set>

/* A single decoded target: either an address (gives hash160) or a pubkey. */
struct cgt_target {
  uint8_t    h160[20];         /* hash160 from address or computed from pubkey */
  bool       fromPubkey;       /* true if specified as a pubkey, false if address */
  bool       compressed;       /* only meaningful if fromPubkey */
  std::string original;        /* the address or pubkey text as given */
};

class CgtPool {
public:
  CgtPool();
  ~CgtPool();

  /* Add a single address (P2PKH mainnet, version 0x00). Returns false on decode failure. */
  bool addAddress(const std::string &addr);
  /* Add a single pubkey (compressed or uncompressed hex). Returns false on parse failure. */
  bool addPubkey(const std::string &pubHex);
  /* Load targets from a text file, one per line, skip empty/comment lines. */
  bool loadFile(const std::string &path, int &loaded, int &failed);

  /* After loading, finalize the bloom filter + exact table. Call once before searching. */
  void finalize(void);

  /* Query: does this hash160 match any target? (bloom + exact check) */
  bool match(const uint8_t h160[20]) const;

  /* Full-precision lookup: returns the matching target, or NULL. Unlike
     match() this compares all 20 bytes, so it never reports a false hit. */
  const cgt_target *lookup(const uint8_t h160[20]) const;

  /* Raw filter, for uploading the identical bits to the GPU. */
  const uint8_t *bloomData(void)  const { return bloom.empty() ? NULL : &bloom[0]; }
  size_t         bloomBytes(void) const { return bloom.size(); }
  int            bloomHashCount(void) const { return bfHashes; }

  /* True if any target was given as a public key rather than an address. */
  bool hasPubkeys(void) const;

  /* Stats */
  size_t count(void) const { return targets.size(); }
  size_t bloomBits(void) const { return bfBits; }
  double bloomFPR(void) const;      /* estimated false positive rate */

private:
  std::vector<cgt_target> targets;
  std::unordered_set<uint64_t> exact;  /* first 8 bytes of each h160 */

  /* Bloom filter: bit-packed array, multiple hash functions */
  std::vector<uint8_t> bloom;
  size_t bfBits;
  int    bfHashes;

  void bloomAdd(const uint8_t h160[20]);
  bool bloomTest(const uint8_t h160[20]) const;
  uint64_t h160Prefix(const uint8_t h160[20]) const;
};

#endif /* CGTPOOL_H */
