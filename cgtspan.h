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
   than a shuffle of a stored list, so the state is a counter and a key - two
   64-bit numbers - no matter how many blocks there are. A 2^80 range is
   1.7e15 blocks; remembering which of those are done as a bitmap would take
   200 TB, and as a permutation counter it takes sixteen bytes.

   The permutation is a four-round Feistel network over 2*half bits with
   cycle-walking down to `count`, which is a bijection for any count. Feistel
   is used rather than a multiply-based mixer because a Feistel round is a
   bijection whatever the round function does, so correctness does not depend
   on the mixer having any particular property - only the apparent randomness
   does.
   --------------------------------------------------------------------------- */
class CgtWalk {
public:
  CgtWalk();

  /* `count` blocks, visited in permuted order when `shuffled`, ascending
     otherwise. `key` selects which permutation; pass a random value for a
     fresh run or the saved one to resume the same order. */
  void init(uint64_t count, uint64_t key, bool shuffled);

  bool     done(void)     const { return pos >= total; }
  uint64_t next(void);            /* block index; advances the cursor */
  uint64_t position(void) const { return pos; }
  uint64_t count(void)    const { return total; }
  uint64_t permKey(void)  const { return key; }
  bool     isShuffled(void) const { return shuffled; }

  /* Resume support: jump the cursor without changing the order. Returns false
     if p is past the end. */
  bool setPosition(uint64_t p);

private:
  uint64_t permute(uint64_t x) const;

  uint64_t total;
  uint64_t key;
  uint64_t pos;
  int      half;        /* bits per Feistel half */
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
  uint64_t blocks;      /* total blocks in the range */
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
