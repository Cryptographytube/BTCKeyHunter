/* cgtcli.cpp - command line front end and search driver for CGTKEY. */
#include "cgtdef.h"
#include "cgtmath.h"
#include "cgtdigest.h"
#include "cgtpool.h"
#include "cgtpkpool.h"
#include "cgtspan.h"
#include "cgtgpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <csignal>
#include <cmath>
#include <string>
#include <vector>
#include <set>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <unistd.h>
#endif

/* Two levels of interrupt. The first asks the driver to stop at the end of the
   current pass, so the checkpoint lands on a block boundary and the summary
   prints normally. A second one means the user is not willing to wait for the
   pass to drain, so we leave immediately. */
static volatile sig_atomic_t cgtStop = 0;

/* Set by the driver once the final checkpoint is on disk. Only the console
   close/logoff/shutdown path reads it - see cgtConsoleHandler. */
static volatile sig_atomic_t cgtDone = 0;

static void cgtOnSignal(int) {
  if (cgtStop) {
    /* Nothing here can safely touch stdio, so just leave. The checkpoint from
       the last completed pass is already on disk. */
    _exit(130);
  }
  cgtStop = 1;
}

#if defined(_WIN32)
/* signal(SIGINT) alone is not enough on Windows. The CRT only maps Ctrl+C, and
   only when the process group has Ctrl+C enabled - a child started with
   CREATE_NEW_PROCESS_GROUP, or one driven from a script, never sees it. Owning
   the console handler directly picks up Ctrl+Break and the window's close box
   as well, which is the difference between a clean stop and losing the pass. */
static BOOL WINAPI cgtConsoleHandler(DWORD type) {
  switch (type) {
  case CTRL_C_EVENT:
  case CTRL_BREAK_EVENT:
    cgtOnSignal(0);
    return TRUE;          /* handled: skip the CRT's default terminator */
  case CTRL_CLOSE_EVENT:
  case CTRL_LOGOFF_EVENT:
  case CTRL_SHUTDOWN_EVENT:
    /* The OS kills us a few seconds from now whatever we return, so ask for a
       stop and hold the handler thread until the driver has written its
       checkpoint. A pass is a fraction of a second; the wait is a safety net,
       not the expected cost. */
    cgtStop = 1;
    for (int i = 0; i < 400 && !cgtDone; i++) Sleep(10);
    return TRUE;
  default:
    break;
  }
  return FALSE;
}
#endif

static double cgtNow(void) {
#if defined(_WIN32)
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart / (double)f.QuadPart;
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/* Width of the last status line written, so it can be blanked before anything
   else prints. Without this the tail of a long status line survives under a
   shorter message and the two run together. */
static int cgtStatusWidth = 0;

static void cgtClearStatus(void) {
  if (cgtStatusWidth <= 0) return;
  printf("\r");
  for (int i = 0; i < cgtStatusWidth; i++) putchar(' ');
  printf("\r");
  cgtStatusWidth = 0;
  fflush(stdout);
}

/* ------------------------------- help screen ------------------------------ */

static void cgtUsage(void) {
  printf("Usage: cgtkey -r <bits> [-a <b58_addr> | -p <pubkey> | -i <file>] [options]\n\n");

  printf("Modes (choose one):\n");
  printf("  -a <b58_addr>       Find the private key for a P2PKH Bitcoin address.\n");
  printf("  -p <pubkey>         Find the private key for a public key (hex, compressed or uncompressed).\n");
  printf("  -i <file>           Search for a list of addresses or public keys from a file (one per line).\n\n");

  printf("Keyspace:\n");
  printf("  -r <bits>           Set the bit range for the search (e.g., 71 for 2^70 to 2^71-1) (required).\n");
  printf("  -r <START:END>      Search a custom hexadecimal range instead of a bit width.\n\n");

  printf("Options:\n");
  printf("  -R                  Activate random mode.\n");
  printf("  -b                  Enable backup mode to resume from last progress.\n");
  printf("  -u                  Search for uncompressed public keys or use uncompressed targets.\n");
  printf("  -G <ID>             Specify the GPU ID to use, default is 0.\n");
  printf("  -h, --help          Display this help message.\n\n");

  printf("Note: When using -i, all targets in the file must be of the same type (all addresses or all public keys).\n");
  printf("      Public-key targets use the fast engine: it compares x-coordinates and never hashes.\n\n");

  printf("%s v.%s\n", CGT_NAME, CGT_RELEASE);
}

/* --------------------------- argument handling ---------------------------- */

static bool cgtNeedValue(int argc, char **argv, int i, const char *what) {
  if (i + 1 < argc && argv[i + 1][0] != '\0') return true;
  fprintf(stderr, "[ERROR] A value is required after the %s parameter.\n", what);
  return false;
}

int main(int argc, char **argv) {
  cgt_config cfg;
  bool haveRange = false;
  bool sawA = false, sawP = false, sawI = false;

  if (argc == 1) { cgtUsage(); return 0; }

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") { cgtUsage(); return 0; }
    else if (a == "-a") {
      if (!cgtNeedValue(argc, argv, i, "-a")) return -1;
      cfg.mode = CGT_MODE_ADDRESS; cfg.single = argv[++i]; sawA = true;
    } else if (a == "-p") {
      if (!cgtNeedValue(argc, argv, i, "-p")) return -1;
      cfg.mode = CGT_MODE_PUBKEY; cfg.single = argv[++i]; sawP = true;
    } else if (a == "-i") {
      if (!cgtNeedValue(argc, argv, i, "-i")) return -1;
      cfg.mode = CGT_MODE_LIST; cfg.listfile = argv[++i]; sawI = true;
    } else if (a == "-r") {
      if (!cgtNeedValue(argc, argv, i, "-r")) return -1;
      cfg.spanText = argv[++i]; haveRange = true;
    } else if (a == "-R") cfg.random = true;
    else if (a == "-u") cfg.uncompressed = true;
    else if (a == "-b") cfg.resume = true;
    else if (a == "-G") {
      if (!cgtNeedValue(argc, argv, i, "-G")) return -1;
      cfg.device = atoi(argv[++i]);
    } else {
      fprintf(stderr, "[ERROR] Unknown parameter: %s\n", a.c_str());
      return -1;
    }
  }

  if ((!sawA && !sawP && !sawI) || !haveRange) {
    fprintf(stderr, "[ERROR] A target source (-a, -p, or -i) and range (-r) must be specified.\n");
    return -1;
  }
  if (sawI && (sawA || sawP)) {
    fprintf(stderr, "[ERROR] Cannot use -i with -a or -p. Please choose one target source.\n");
    return -1;
  }
  if (sawA && sawP) {
    fprintf(stderr, "[ERROR] Cannot use -a and -p at the same time. Please choose one.\n");
    return -1;
  }

  /* ------------------------------ keyspace ------------------------------- */
  cgt_u256 ksStart, ksEnd;
  if (!cgtParseRange(cfg.spanText, ksStart, ksEnd)) {
    fprintf(stderr, "[ERROR] Invalid -r value. Expected a bit count 1..256 or START:END in hex.\n");
    return -1;
  }

  /* ------------------------------- targets -------------------------------
     Two engines, two target pools. When every target is a public key the x
     coordinate is itself the thing being searched for, so the pubkey pool is
     filled and the hash-free kernel runs. Anything involving an address needs
     hash160, which means the address pool and the hashing kernel. */
  CgtPool pool;
  CgtPubkeyPool pkpool;
  bool pkMode = false;
  std::string label;
  if (cfg.mode == CGT_MODE_LIST) {
    int loaded = 0, failed = 0;
    if (!pool.loadFile(cfg.listfile, loaded, failed)) {
      fprintf(stderr, "[ERROR] Could not open target file: %s\n", cfg.listfile.c_str());
      return -1;
    }
    if (loaded == 0) {
      fprintf(stderr, "[ERROR] Target file '%s' is empty or contains no valid lines.\n",
              cfg.listfile.c_str());
      return -1;
    }
    printf("[+] Targets loaded from file: %d\n", loaded);
    if (failed) printf("[+] Lines skipped (unparsable): %d\n", failed);
    label = pool.hasPubkeys() ? "[Public Keys]" : "[P2PKH]";

    /* A pubkey-only file can take the fast engine. Re-reading the file into the
       pubkey pool is a one-off startup cost and keeps the two pools independent.
       Counts differing means the file mixes addresses with pubkeys, and an
       address can only be matched through hash160 - so that falls back. */
    if (pool.hasPubkeys()) {
      int pkLoaded = 0, pkFailed = 0;
      if (pkpool.loadFile(cfg.listfile, pkLoaded, pkFailed) &&
          pkLoaded == loaded && pkpool.count() == pool.count()) {
        pkMode = true;
      } else {
        printf("[+] File mixes addresses and public keys: using the address engine.\n");
      }
    }
  } else if (cfg.mode == CGT_MODE_ADDRESS) {
    if (!pool.addAddress(cfg.single)) {
      fprintf(stderr, "[ERROR] '%s' is not a valid P2PKH address.\n", cfg.single.c_str());
      return -1;
    }
    label = "[P2PKH/Compressed]";
  } else {
    if (!pool.addPubkey(cfg.single)) {
      fprintf(stderr, "[ERROR] '%s' is not a valid public key.\n", cfg.single.c_str());
      return -1;
    }
    if (!pkpool.addPubkey(cfg.single)) {
      fprintf(stderr, "[ERROR] '%s' is not a valid public key.\n", cfg.single.c_str());
      return -1;
    }
    pkMode = true;
    label = "[Public Key]";
  }
  pool.finalize();
  if (pkMode) pkpool.finalize();

  /* -u selects the serialisation to hash, so it only means anything to the
     address engine. The pubkey engine compares x, which both serialisations
     share, so it finds a key given either form of the same point - and the
     hit's format is reported from what the target itself declared. */
  if (pkMode && cfg.uncompressed)
    printf("[+] Pubkey mode compares x-coordinates: -u has no effect.\n");

  /* --------------------------------- GPU --------------------------------- */
  std::vector<cgt_devinfo> devs;
  if (!cgtGpuList(devs)) {
    fprintf(stdout, "[INFO] No CUDA-enabled GPU was detected. Exiting.\n");
    return -1;
  }
  if (cfg.device < 0 || cfg.device >= (int)devs.size()) {
    fprintf(stdout, "[INFO] Invalid GPU ID %d specified. No device detected with this ID.\n", cfg.device);
    fprintf(stdout, "[INFO] Detected %d GPU(s). Valid IDs are from 0 to %d.\n",
            (int)devs.size(), (int)devs.size() - 1);
    return -1;
  }

  CgtGpu gpu;
  std::string err;
  if (!gpu.open(cfg.device, err)) {
    fprintf(stderr, "[ERROR] GPU init failed: %s\n", err.c_str());
    return -1;
  }
  if (pkMode) {
    /* One x per point, so the kernel needs exactly one comparison per key -
       the COMP specialisation - regardless of how the targets were written. */
    gpu.setModes(true, false);
    if (!gpu.uploadPubkeyFilter(pkpool.bloomData(), pkpool.bloomBytes(),
                                (uint64_t)pkpool.bloomBits(), pkpool.bloomHashCount(),
                                pkpool.xCoordData(), pkpool.xCoordCount(), err)) {
      fprintf(stderr, "[ERROR] Pubkey filter upload failed: %s\n", err.c_str());
      return -1;
    }
  } else {
    gpu.setModes(!cfg.uncompressed, cfg.uncompressed);
    if (!gpu.uploadFilter(pool.bloomData(), pool.bloomBytes(),
                          (uint64_t)pool.bloomBits(), pool.bloomHashCount(), err)) {
      fprintf(stderr, "[ERROR] Filter upload failed: %s\n", err.c_str());
      return -1;
    }
  }

  const int lanes = gpu.laneCount();
  const uint64_t stride = (uint64_t)lanes * (uint64_t)CGT_LANE_REGION;

  /* ---------------------------- the block walk ---------------------------
     Cut the range into `stride`-key blocks. One block is one GPU pass, and the
     walk hands out every block index exactly once. Ascending in normal mode,
     permuted in random mode - either way the run ends only when the range has
     been covered in full, with nothing repeated and nothing skipped. */
  cgt_u256 spanU;
  cgtSub(spanU, ksEnd, ksStart);            /* inclusive width is spanU + 1 */
  double spanKeys = cgtToDouble(spanU) + 1.0;

  uint64_t blockCount;
  {
    /* ceil((span+1)/stride) without risking the +1 overflowing a full range:
       floor(span/stride) + 1 is the same number for every span. */
    cgt_u256 q;
    cgtDivU64(q, spanU, stride);
    if (q.w[1] | q.w[2] | q.w[3]) blockCount = ~0ULL;   /* astronomically large */
    else if (q.w[0] == ~0ULL)     blockCount = ~0ULL;
    else                          blockCount = q.w[0] + 1ULL;
  }

  cgtSeedRNG();
  cgtBuildLadder();

  CgtWalk walk;
  CgtCheckpoint ckpt;
  uint64_t priorFound = 0;
  bool restored = false;

  if (cfg.resume) {
    cgt_resume st;
    if (ckpt.load(st)) {
      /* A checkpoint is only meaningful against the range and block size it
         was written for. A different -r, or a different GPU (which changes the
         lane count and therefore the block size), would make the saved cursor
         point somewhere else entirely - so say so rather than resume wrongly. */
      if (cgtCmp(st.start, ksStart) != 0 || cgtCmp(st.end, ksEnd) != 0) {
        printf("[+] Backup file is for a different range. Starting from scratch.\n");
      } else if (st.stride != stride || st.blocks != blockCount) {
        printf("[+] Backup file was written with a different GPU geometry. Starting from scratch.\n");
      } else if (st.shuffled != cfg.random) {
        printf("[+] Backup file was written in %s mode. Starting from scratch.\n",
               st.shuffled ? "random" : "sequential");
      } else {
        walk.init(blockCount, st.key, cfg.random);
        walk.setPosition(st.pos);
        priorFound = st.found;
        restored = true;
      }
    } else {
      printf("[+] Backup file not found. Will start from scratch.\n");
    }
  }
  if (!restored) walk.init(blockCount, cgtRand64(), cfg.random);

  /* -------------------------------- banner ------------------------------- */
  const cgt_devinfo &d = gpu.info();
  time_t now = time(NULL);

  printf("\n[+] %s v.%s\n", CGT_NAME, CGT_RELEASE);
  if (cfg.mode == CGT_MODE_LIST)
    printf("[+] Search: %zu targets %s\n", pool.count(), label.c_str());
  else
    printf("[+] Search: %s %s\n", cfg.single.c_str(), label.c_str());
  printf("[+] Start %s", ctime(&now));
  if (cfg.random) printf("[+] Random mode\n");
  if (cfg.spanText.find(':') != std::string::npos)
    printf("[+] Range (Custom Hex)\n");
  else
    printf("[+] Range (2^%s)\n", cfg.spanText.c_str());
  printf("[+] from : 0x%s\n", cgtToHex256(ksStart).c_str());
  printf("[+] to   : 0x%s\n", cgtToHex256(ksEnd).c_str());
  printf("[+] GPU %d: %s (%d SM, cap %d.%d, %.0f MB)\n", d.id, d.name.c_str(),
         d.smCount, d.major, d.minor, (double)d.memBytes / (1024.0 * 1024.0));
  printf("[+] Lanes %d, %s keys per pass\n", lanes,
         cgtScaleCount((double)gpu.keysPerPass()).c_str());
  printf("[+] Coverage: %llu blocks of %s keys, each visited once\n",
         (unsigned long long)blockCount,
         cgtScaleCount((double)stride).c_str());
  if (pkMode)
    printf("[+] Engine: pubkey (x-coordinate compare, no SHA-256/RIPEMD-160)\n");
  else
    printf("[+] Engine: address (hash160)\n");
  if (cfg.uncompressed && !pkMode) printf("[+] Uncompressed targets\n");
  if (restored) {
    double pct = blockCount ? (double)walk.position() * 100.0 / (double)blockCount : 0.0;
    printf("[+] Restoring from backup was successful. Resuming at block %llu of %llu (%.4f%% done)\n",
           (unsigned long long)walk.position(), (unsigned long long)blockCount, pct);
  }
  printf("\n");

  /* ------------------------------ main loop ------------------------------ */
  signal(SIGINT, cgtOnSignal);
#if defined(SIGTERM)
  signal(SIGTERM, cgtOnSignal);
#endif
#if defined(_WIN32)
  SetConsoleCtrlHandler(cgtConsoleHandler, TRUE);
#endif

  /* A lane's anchor sits in the middle of the window it sweeps. */
  cgt_u256 halfSpan;
  cgtSetU64(halfSpan, CGT_STRIDE_HALF);

  cgt_u256 laneRegion;
  cgtSetU64(laneRegion, (uint64_t)CGT_LANE_REGION);

  std::vector<cgt_hit> hits;
  std::set<std::string> seenKeys;     /* so one key is reported once */

  double t0 = cgtNow();
  double lastReport = t0;
  double lastCkpt = t0;
  double totalKeys = 0.0;
  uint64_t found = priorFound;

  /* Set CGTKEY_PROFILE=1 to break each pass into host-setup vs GPU time.
     Not advertised in the help screen; it exists for tuning. */
  const bool profile = getenv("CGTKEY_PROFILE") != NULL;
  double accSetup = 0.0, accGpu = 0.0;

  cgt_u256 blockBase;                 /* first key of the block being scanned */
  cgtSetZero(blockBase);
  uint64_t prevBlock = 0;
  bool havePrev = false;
  std::string lastKeyText = "-";
  uint64_t passCounter = 0;           /* seeds the status line's sampled key */

  while (!cgtStop && !walk.done()) {
    double tPassStart = profile ? cgtNow() : 0.0;

    uint64_t blk = walk.next();

    /* blockBase = ksStart + blk*stride */
    {
      cgt_u256 off;
      cgtSetU64(off, blk);
      cgt_u256 scaled;
      cgtMulU64(scaled, off, stride);
      cgtAdd(blockBase, ksStart, scaled);
    }

    /* Seeding costs one host scalar multiply. When the walk happens to hand
       out the very next block - which is every pass in sequential mode - the
       device can slide its own anchors forward instead, and the host does no
       curve work at all. */
    if (havePrev && blk == prevBlock + 1) {
      if (!gpu.advanceAnchors(err)) {
        cgtClearStatus();
        fprintf(stderr, "[ERROR] %s\n", err.c_str());
        return -1;
      }
    } else {
      cgt_u256 seed;
      cgtAdd(seed, blockBase, halfSpan);
      if (!gpu.seedAnchorsRandom(seed, err)) {
        cgtClearStatus();
        fprintf(stderr, "[ERROR] %s\n", err.c_str());
        return -1;
      }
    }
    prevBlock = blk;
    havePrev = true;

    double tSetup = profile ? cgtNow() : 0.0;

    hits.clear();
    if (!gpu.runPass(hits, err)) {
      cgtClearStatus();
      fprintf(stderr, "[ERROR] %s\n", err.c_str());
      return -1;
    }
    double tGpu = profile ? cgtNow() : 0.0;

    if (profile) {
      accSetup += (tSetup - tPassStart);
      accGpu += (tGpu - tSetup);
    }

    totalKeys += (double)gpu.keysPerPass();

    /* A key from the pass that just finished, shown so the run is visibly
       moving and it is obvious where in the range it currently is.

       Naming the block's last key (blockBase + stride - 1) would be correct but
       useless to watch: blocks are stride-aligned and stride is 0x23046000, so
       the low three hex digits came out FFF on literally every line. The pass
       checks every key in [blockBase, blockBase + stride), so any offset in
       that span is an equally real answer to "what was just checked" - pick one
       by hashing the pass counter, which keeps every digit moving. */
    {
      uint64_t m = passCounter * 0x9E3779B97F4A7C15ULL;
      m ^= m >> 29; m *= 0xBF58476D1CE4E5B9ULL; m ^= m >> 32;
      cgt_u256 last;
      cgtAddU64(last, blockBase, m % stride);
      lastKeyText = cgtToHex256(last);
      passCounter++;
    }

    /* --- confirm candidates on the host --- */
    for (size_t i = 0; i < hits.size(); i++) {
      const cgt_hit &h = hits[i];
      if (h.lane >= (uint32_t)lanes) continue;

      /* Address mode screens on the digest the device sent back, which throws
         out a bloom false positive before paying for a scalar multiply. Pubkey
         mode cannot: the hit carries only 160 bits of a 256-bit x, so the key
         has to be rebuilt and the point recomputed to know what was really
         found. Hits are rare either way. */
      const cgt_target *t = NULL;
      if (!pkMode) {
        /* RIPEMD-160 emits its state as little-endian words, and the device
           bloom hashes the same little-endian bytes, so unpack that way. */
        uint8_t dig[20];
        for (int j = 0; j < CGT_H160_W; j++) {
          dig[4*j]     = (uint8_t)(h.h160[j]);
          dig[4*j + 1] = (uint8_t)(h.h160[j] >> 8);
          dig[4*j + 2] = (uint8_t)(h.h160[j] >> 16);
          dig[4*j + 3] = (uint8_t)(h.h160[j] >> 24);
        }
        t = pool.lookup(dig);
        if (!t) continue;               /* bloom false positive */
      }

      /* Rebuild the exact private key. Lane l's anchor for this block is
         blockBase + HALF + l*CGT_LANE_REGION, and the step code carries the
         signed distance from there. */
      cgt_u256 laneOff;
      {
        cgt_u256 l;
        cgtSetU64(l, (uint64_t)h.lane);
        cgtMulU64(laneOff, l, (uint64_t)CGT_LANE_REGION);
      }
      cgt_u256 anchorNow;
      cgtAdd(anchorNow, blockBase, halfSpan);
      {
        cgt_u256 tmp;
        cgtAdd(tmp, anchorNow, laneOff);
        anchorNow = tmp;
      }

      cgt_u256 delta;
      bool neg = false;
      CgtGpu::offsetForStep(h.step, delta, neg);
      cgt_u256 key;
      if (neg) cgtSub(key, anchorNow, delta);
      else     cgtAdd(key, anchorNow, delta);

      if (pkMode) {
        cgt_pt p;
        cgtPtMulG(p, key);

        const cgt_pktarget *pt = pkpool.lookup(p.x);
        if (!pt) continue;              /* bloom false positive */

        /* x matched, but P and -P share an x. If y disagrees the device found
           the mirror, and the scalar belonging to the target is n - key. */
        if (cgtCmp(p.y, pt->y) != 0) {
          cgt_u256 mirror;
          cgtScalarNeg(mirror, key);
          cgt_pt q;
          cgtPtMulG(q, mirror);
          if (cgtCmp(q.x, pt->x) != 0 || cgtCmp(q.y, pt->y) != 0) continue;
          key = mirror;
          p = q;
        }

        bool comp = pt->compressed;
        std::string kh = cgtToHex256Pad(key);
        if (!seenKeys.insert(kh + (comp ? "c" : "u")).second) continue;

        found++;
        std::vector<uint8_t> ser;
        cgtSerializePub(p, comp, ser);
        uint8_t hp[20];
        cgtHash160(&ser[0], ser.size(), hp);
        std::string addr = cgtAddrFromHash160(hp);

        cgtClearStatus();
        printf("\n=======================================================\n");
        printf("  KEY FOUND\n");
        printf("  Target  : %s\n", pt->original.c_str());
        printf("  Address : %s\n", addr.c_str());
        printf("  PrivKey : 0x%s\n", kh.c_str());
        printf("  Format  : %s\n", comp ? "compressed" : "uncompressed");
        printf("=======================================================\n\n");
        fflush(stdout);

        FILE *outp = fopen("cgtkey_found.txt", "a");
        if (outp) {
          fprintf(outp, "Target : %s\nAddress: %s\nPrivKey: 0x%s\nFormat : %s\n\n",
                  pt->original.c_str(), addr.c_str(), kh.c_str(),
                  comp ? "compressed" : "uncompressed");
          fclose(outp);
        }
        continue;
      }

      /* Independently verify on the host before claiming a hit. */
      bool comp = (h.flags & 2u) == 0u;
      cgt_pt p;
      cgtPtMulG(p, key);
      std::vector<uint8_t> ser;
      cgtSerializePub(p, comp, ser);
      uint8_t check[20];
      cgtHash160(&ser[0], ser.size(), check);
      if (memcmp(check, t->h160, 20) != 0) continue;

      std::string kh = cgtToHex256Pad(key);
      /* The same key can surface twice - a target listed twice in the file,
         or the compressed and uncompressed forms of one point when both
         flavours are enabled. Report it once. */
      if (!seenKeys.insert(kh + (comp ? "c" : "u")).second) continue;

      found++;
      std::string addr = cgtAddrFromHash160(check);
      cgtClearStatus();
      printf("\n=======================================================\n");
      printf("  KEY FOUND\n");
      printf("  Target  : %s\n", t->original.c_str());
      printf("  Address : %s\n", addr.c_str());
      printf("  PrivKey : 0x%s\n", kh.c_str());
      printf("  Format  : %s\n", comp ? "compressed" : "uncompressed");
      printf("=======================================================\n\n");
      fflush(stdout);

      FILE *out = fopen("cgtkey_found.txt", "a");
      if (out) {
        fprintf(out, "Target : %s\nAddress: %s\nPrivKey: 0x%s\nFormat : %s\n\n",
                t->original.c_str(), addr.c_str(), kh.c_str(),
                comp ? "compressed" : "uncompressed");
        fclose(out);
      }
    }

    /* --- periodic status line --- */
    double now2 = cgtNow();
    if (now2 - lastReport >= 1.0) {
      double el = now2 - t0;
      double rate = el > 0.0 ? totalKeys / el : 0.0;

      /* Progress is the walk's own cursor, so it counts blocks actually
         retired rather than an estimate - and a resumed run picks up the
         percentage it left off at. */
      double done = (double)walk.position() * (double)stride;
      double prob = spanKeys > 0.0 ? (done / spanKeys) * 100.0 : 0.0;
      if (prob > 100.0) prob = 100.0;
      double half = (rate > 0.0) ? (spanKeys * 0.5 - done) / rate / 31557600.0 : 0.0;
      if (half < 0.0) half = 0.0;

      char line[320];
      if (profile && accSetup > 0.0 && accGpu > 0.0) {
        snprintf(line, sizeof(line),
                 "[GPU %.2f Mkey/s][Setup %.2f%%][Total 2^%.2f][Prob %.1e%%][Found %llu][%s]",
                 rate / 1.0e6,
                 (accSetup / (accSetup + accGpu)) * 100.0,
                 totalKeys > 0.0 ? log(totalKeys) / log(2.0) : 0.0,
                 prob, (unsigned long long)found, lastKeyText.c_str());
      } else {
        snprintf(line, sizeof(line),
                 "[GPU %.2f Mkey/s][Total 2^%.2f][Prob %.1e%%][50%% in %s][Found %llu][%s]",
                 rate / 1.0e6,
                 totalKeys > 0.0 ? log(totalKeys) / log(2.0) : 0.0,
                 prob, cgtHumanYears(half).c_str(),
                 (unsigned long long)found, lastKeyText.c_str());
      }

      /* Pad out to the previous line's width so a shorter line cannot leave
         characters from the longer one behind it. */
      int len = (int)strlen(line);
      printf("\r%s", line);
      for (int i = len; i < cgtStatusWidth; i++) putchar(' ');
      cgtStatusWidth = len;
      fflush(stdout);
      lastReport = now2;
    }

    /* --- checkpoint --- */
    if (cfg.resume && now2 - lastCkpt >= 10.0) {
      cgt_resume st;
      st.start = ksStart; st.end = ksEnd;
      st.stride = stride; st.blocks = blockCount;
      st.key = walk.permKey(); st.pos = walk.position();
      st.shuffled = walk.isShuffled(); st.found = found;
      ckpt.save(st);
      lastCkpt = now2;
    }
  }

  bool exhausted = walk.done();

  /* Final checkpoint on the way out, so a Ctrl+C loses at most the pass that
     was in flight - never any block the walk already retired. */
  if (cfg.resume) {
    cgt_resume st;
    st.start = ksStart; st.end = ksEnd;
    st.stride = stride; st.blocks = blockCount;
    st.key = walk.permKey(); st.pos = walk.position();
    st.shuffled = walk.isShuffled(); st.found = found;
    ckpt.save(st);
  }

  cgtClearStatus();
  if (cgtStop) printf("\n[+] Stopping at the end of the current pass.\n");

  double el = cgtNow() - t0;
  printf("\n[+] Scanned %s keys in %.1f s (%.2f Mkey/s)\n",
         cgtScaleCount(totalKeys).c_str(), el,
         el > 0.0 ? totalKeys / el / 1.0e6 : 0.0);
  printf("[+] Progress: block %llu of %llu (%.6f%% of the range)\n",
         (unsigned long long)walk.position(), (unsigned long long)blockCount,
         blockCount ? (double)walk.position() * 100.0 / (double)blockCount : 0.0);
  if (exhausted) {
    printf("[+] Range fully covered: every key in the range was checked exactly once.\n");
    if (found == 0) printf("[+] Range exhausted, no match found.\n");
  } else if (cfg.resume) {
    printf("[+] Progress saved to %s. Re-run with -b to continue from here.\n",
           ckpt.file().c_str());
  } else if (cgtStop) {
    printf("[+] Progress not saved (run with -b to make a stopped run resumable).\n");
  }
  printf("[+] Found %llu key%s\n", (unsigned long long)found, found == 1 ? "" : "s");
  fflush(stdout);
  cgtDone = 1;     /* releases a console close/shutdown handler that is waiting */
  return found > 0 ? 0 : 1;
}
