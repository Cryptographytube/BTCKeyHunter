/* cgtgpu.h - CUDA search engine interface. */
#ifndef CGTGPU_H
#define CGTGPU_H

#include "cgtdef.h"
#include "cgtmath.h"
#include <string>
#include <vector>

/* Device inventory entry, used by the banner and by -G validation. */
struct cgt_devinfo {
  int         id;
  std::string name;
  int         major, minor;      /* compute capability */
  int         smCount;
  int         coresPerSM;
  size_t      memBytes;
};

/* Enumerate visible CUDA devices. Returns false if CUDA is unavailable. */
bool cgtGpuList(std::vector<cgt_devinfo> &out);

class CgtGpu {
public:
  CgtGpu();
  ~CgtGpu();

  /* Bring up device `id` and size the launch grid. Returns false on failure. */
  bool open(int id, std::string &err);

  /* Upload the bloom filter so the kernel can pre-screen hashes on-device.
     bits/hashes must match the host filter exactly. */
  bool uploadFilter(const uint8_t *bloom, size_t bloomBytes,
                    uint64_t bits, int hashes, std::string &err);

  /* Pubkey engine: upload the x-coordinate bloom plus the exact target list.
     Selecting this instead of uploadFilter switches runPass() over to the
     hash-free kernel, which compares x-coordinates directly. `xCoords` is
     4 limbs per target, little-endian, exactly as CgtPubkeyPool stores them. */
  bool uploadPubkeyFilter(const uint8_t *bloom, size_t bloomBytes,
                          uint64_t bits, int hashes,
                          const uint64_t *xCoords, size_t targetCount,
                          std::string &err);

  /* True once uploadPubkeyFilter has put the engine in hash-free mode. */
  bool pubkeyMode(void) const { return pkMode; }

  /* Choose which digest flavours the kernel emits. */
  void setModes(bool compressed, bool uncompressed);

  /* Seed the lane anchors from host-computed starting points. Called once at
     the beginning; thereafter anchors live on device and slide forward via
     advanceAnchors(). `keys` must hold exactly laneCount() entries. */
  bool seedAnchors(const std::vector<cgt_u256> &keys, std::string &err);

  /* Random mode: place lane i's anchor at scalar base + i*CGT_LANE_REGION,
     using one host scalar multiply plus a device kernel. Deriving all
     laneCount() anchors from scalars on the host cost more than the search
     itself. */
  bool seedAnchorsRandom(const cgt_u256 &base, std::string &err);

  /* Slide each lane's anchor forward by one region (on-device). Much cheaper
     than recomputing from scalars on the host. */
  bool advanceAnchors(std::string &err);

  /* Run one pass over every lane. Appends any bloom hits to `hits`. */
  bool runPass(std::vector<cgt_hit> &hits, std::string &err);

  /* Grid geometry. */
  int      laneCount(void)   const { return lanes; }
  uint64_t keysPerPass(void) const;
  const cgt_devinfo &info(void) const { return dev; }

  /* Distance between a lane anchor and a given step index, so the host can
     reconstruct the exact private key from a hit. */
  static void offsetForStep(uint32_t step, cgt_u256 &delta, bool &negative);

private:
  cgt_devinfo dev;
  int  lanes;
  int  blocks;
  int  threadsPerBlock;
  bool wantComp, wantUncomp;
  bool ready;

  /* device buffers */
  void     *dAnchorX, *dAnchorY;   /* per-lane starting point      */
  void     *dBloom;                /* bloom bit array              */
  void     *dTargets;              /* x-coords for pubkey engine   */
  void     *dPre;                  /* stage-one screen, -> shared  */
  int       targetCount;           /* how many x-coords uploaded   */
  bool      pkMode;                /* true = pubkey, false = addr  */
  void     *dHits;                 /* cgt_hit[CGT_HIT_SLOTS]       */
  void     *dHitCount;             /* uint32 counter               */
  void     *dStepX, *dStepY;       /* precomputed i*G table        */
  void     *dJumpX, *dJumpY;       /* CGT_LANE_KEYS * G            */
  void     *dGridX, *dGridY;       /* lanes*CGT_LANE_REGION * G    */
  void     *dLaneOffX, *dLaneOffY; /* i*CGT_LANE_REGION * G, per lane */
  void     *dBaseX, *dBaseY;       /* random-mode base point       */
  uint64_t  bloomBits;
  int       bloomHashes;
};

#endif /* CGTGPU_H */
