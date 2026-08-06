/* cgtdef.h - CGTKEY common definitions, limb types and tunables. */
#ifndef CGTDEF_H
#define CGTDEF_H

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#define CGT_NAME     "CGTKEY"
#define CGT_RELEASE  "1.0"

/* --- 256-bit little-endian limb bundle used on host and device. --- */
#define CGT_LIMBS 4
typedef struct { uint64_t w[CGT_LIMBS]; } cgt_u256;

/* hash160 digest, 20 bytes viewed as 5 x uint32 */
#define CGT_H160_W 5

/* --- Search shape tunables -------------------------------------------- */
/* These three set the geometry of the search and trade off against each
   other, so they are left overridable with -D for benchmarking rather than
   baked in. The defaults are what measured fastest on the hardware to hand;
   a different card may prefer a different corner. */

/* Table entries per lane: the lane visits anchor +/- i*G for i in
   [1, CGT_STRIDE_HALF], so it yields 2*CGT_STRIDE_HALF keys per pass.
   P+Q and P-Q share the denominator (x2-x1), so one modular inverse
   serves both signs.

   This is the single most important tunable, because it sets how far the batch
   inversion is amortized. The inversion costs ~270 field multiplies however
   many points share it, so at 64 entries it adds ~2 muls per key and at 1024
   it adds ~0.13. The cost is per-thread scratch - one field element per entry,
   32 KiB at 1024 - which is why pushing to 2048 measured slower again. */
#ifndef CGT_STRIDE_HALF
#define CGT_STRIDE_HALF 1024
#endif
#define CGT_LANE_POINTS (CGT_STRIDE_HALF * 2)

/* Tuning note, recorded so it is not re-explored: the batch inversion's forward
   and backward passes are each one dependency chain of back-to-back 256-bit
   multiplies, and raising occupancy from 8 to 16 warps per SM changes throughput
   not at all - so the kernel is chain-bound, not occupancy-bound. Splitting the
   running product across 2 or 4 independent accumulators to overlap those chains
   was implemented and measured, and is slower at every geometry that fits: each
   accumulator adds ~16 registers to an already-full register file (128 regs x
   512 threads = 65536 exactly), so fitting needs either -maxrregcount (spills,
   3570 Mkey/s) or a half-size block (3047 Mkey/s), against 3867 for the single
   accumulator. Register capacity, not chain length, is the binding constraint. */

/* Keys actually covered by one lane in one round: the +/- walk plus the anchor
   itself, i.e. the closed window [anchor - HALF, anchor + HALF]. */
#define CGT_LANE_KEYS (CGT_LANE_POINTS + 1)

/* Rounds walked inside a single kernel launch. After each round the lane
   advances its anchor by CGT_LANE_KEYS via one more point addition, whose
   denominator rides along in the same batch inversion - so the advance is
   nearly free. Raising this amortizes the host-side setup over more GPU work,
   which stopped mattering once the anchors became resident on the device. */
#ifndef CGT_ROUNDS
#define CGT_ROUNDS 8
#endif

/* Contiguous keys owned by one lane for a whole launch. */
#define CGT_LANE_REGION (CGT_ROUNDS * CGT_LANE_KEYS)

/* Threads per CUDA block. */
#ifndef CGT_BLOCK_THREADS
#define CGT_BLOCK_THREADS 512
#endif

/* Blocks launched per SM. Together with CGT_BLOCK_THREADS this decides
   occupancy, which the register count caps from the other side.

   One big block per SM beat two smaller ones by ~15%. The kernel is not
   latency-bound in the usual way - each lane runs a long dependent chain of
   field multiplies over its own private scratch - so extra concurrent blocks
   buy little overlap while doubling the local-memory footprint the SM has to
   keep resident. */
#ifndef CGT_BLOCKS_PER_SM
#define CGT_BLOCKS_PER_SM 1
#endif

/* Result slots handed back by one kernel launch. */
#define CGT_HIT_SLOTS 4096

/* Step encoding inside a launch: step = round * CGT_LANE_KEYS + local, where
   local in [0, CGT_LANE_POINTS-1] is the +/- walk (even = +, odd = -) and
   local == CGT_STEP_ANCHOR_OFF marks the round's anchor itself (delta 0). */
#define CGT_STEP_ANCHOR_OFF ((uint32_t)CGT_LANE_POINTS)

/* One hit record: lane index, step offset inside the lane, parity, digest. */
typedef struct {
  uint32_t lane;
  uint32_t step;      /* 0 .. CGT_LANE_POINTS-1 */
  uint32_t flags;     /* bit0 = negated (mirror) point, bit1 = uncompressed */
  uint32_t h160[CGT_H160_W];
} cgt_hit;

/* Search flavour requested on the command line. */
enum cgt_mode {
  CGT_MODE_ADDRESS = 0,   /* single P2PKH address        */
  CGT_MODE_PUBKEY  = 1,   /* explicit compressed pubkey  */
  CGT_MODE_LIST    = 2    /* many targets from a file    */
};

/* Runtime knobs shared between the CLI and the engine. */
struct cgt_config {
  cgt_mode      mode;
  std::string   single;        /* address or pubkey text  */
  std::string   listfile;      /* -i file                 */
  std::string   spanText;      /* raw -r argument         */
  bool          random;        /* -R                      */
  bool          uncompressed;  /* -u                      */
  bool          resume;        /* -b                      */
  int           device;        /* -G                      */
  cgt_config()
    : mode(CGT_MODE_ADDRESS), random(false), uncompressed(false),
      resume(false), device(0) {}
};

/* Small helpers shared by several translation units. */
std::string cgtTrim(const std::string &s);
std::string cgtUpper(const std::string &s);
bool        cgtHexToBytes(const std::string &hex, std::vector<uint8_t> &out);
std::string cgtBytesToHex(const uint8_t *p, size_t n);
std::string cgtScaleCount(double v);      /* 1234567 -> "1.23 M" */
std::string cgtHumanYears(double years);

#endif /* CGTDEF_H */
