/* cgtspan.h - keyspace range parsing, block walk order, resume/backup. */
#ifndef CGTSPAN_H
#define CGTSPAN_H

#include "cgtdef.h"
#include "cgtmath.h"
#include <string>

/* Parse the -r argument into a start..end range.
   - Single decimal bit count: "66" => [2^65, 2^66-1]
   - Hex range: "START:END" => [START, END] inclusive */
bool cgtParseRange(const std::string &txt, cgt_u256 &start, cgt_u256 &end);

/* Random 256-bit scalar in [lo, hi] inclusive, uniform if hi-lo is not huge. */
void cgtRand256(cgt_u256 &out, const cgt_u256 &lo, const cgt_u256 &hi);

/* A fresh 64-bit value from the seeded RNG. */
uint64_t cgtRand64(void);

/* Initialize the RNG from system entropy. Call once at startup. */
void cgtSeedRNG(void);

/* ---------------------------------------------------------------------------
   CgtWalk - the order in which the search visits the range.

   The range is cut into equal blocks of `stride` keys, one block per GPU pass.
   The walk hands out every block index in [0, count) exactly once and then
   reports itself done. That is what makes a run exhaustive: no block can be
   skipped, none can come up twice, and when the walk is finished the range has
   genuinely been covered.

   In shuffled (random) mode the order is a pseudo-random permutation rather
   than a shuffle of a stored list, so the state is a counter and a key - no
   matter how many blocks there are. A 2^80 range is 1.7e15 blocks; remembering
   which of those are done as a bitmap would take 200 TB, and as a permutation
   counter it takes a handful of bytes.

   The permutation is a four-round Feistel network over 2*half bits (half up to
   128, so the domain spans the full 256-bit block count) with cycle-walking
   down to `count`, which is a bijection for any count. Feistel is used rather
   than a multiply-based mixer because a Feistel round is a bijection whatever
   the round function does, so correctness does not depend on the mixer having
   any particular property - only the apparent randomness does.

   `count` is a full 256-bit value: a bit range up to 2^256 cuts into up to
   ~2^226 blocks, and the block index is fed straight into blockBase =
   ksStart + blk*stride, so EVERY bit of the base varies from block to block -
   the high bits included. (A 64-bit index times a ~2^30 stride could only
   reach ksStart + 2^94, which is what used to leave the top of a big range
   frozen at the start value.) The cursor `pos` stays 64-bit: no run retires
   2^64 blocks, and in random mode those 2^64 draws still land anywhere in the
   256-bit index space via the permutation, so the sampled bases still cover
   the whole range with the high bits fully random.
   --------------------------------------------------------------------------- */
class CgtWalk {
public:
  CgtWalk();

  /* `count` blocks (up to 2^256), visited in permuted order when `shuffled`,
     ascending otherwise. `key` selects which permutation; pass a random value
     for a fresh run or the saved one to resume the same order. */
  void init(const cgt_u256 &count, uint64_t key, bool shuffled);

  bool     done(void) const;                    /* pos >= total (256-bit)      */
  cgt_u256 next(void);                           /* block index; advances cursor */
  uint64_t position(void) const { return pos; }
  const cgt_u256 &count(void) const { return total; }
  uint64_t permKey(void)  const { return key; }
  bool     isShuffled(void) const { return shuffled; }

  /* Resume support: jump the cursor without changing the order. Returns false
     if p is past the end. */
  bool setPosition(uint64_t p);

private:
  void permute(const cgt_u256 &x, cgt_u256 &out) const;

  cgt_u256 total;       /* block count, up to 2^256                            */
  uint64_t key;
  uint64_t pos;         /* blocks retired so far (a run never reaches 2^64)    */
  int      half;        /* bits per Feistel half, 1..128                       */
  bool     shuffled;
};

/* ---------------------------------------------------------------------------
   CgtCheckpoint - what -b writes to disk.

   It stores the walk's cursor, not the keys. Sixteen bytes of state stand in
   for however many keys have been checked, so stopping and restarting a run
   picks up exactly where it left off and never re-covers ground. The range and
   block size are recorded alongside so a resume against a different range - or
   a different GPU, which changes the block size - is detected and refused
   rather than silently resuming into the wrong place.
   --------------------------------------------------------------------------- */
struct cgt_resume {
  cgt_u256 start, end;
  uint64_t stride;      /* keys per block */
  cgt_u256 blocks;      /* total blocks in the range (up to 2^256) */
  uint64_t key;         /* permutation key */
  uint64_t pos;         /* blocks already completed */
  bool     shuffled;    /* was it a random-order run */
  uint64_t found;       /* keys found so far */
};

class CgtCheckpoint {
public:
  CgtCheckpoint();
  ~CgtCheckpoint();

  void setPath(const std::string &p) { path = p; }
  const std::string &file(void) const { return path; }

  bool load(cgt_resume &st);
  bool save(const cgt_resume &st);

private:
  std::string path;
};

#endif /* CGTSPAN_H */
