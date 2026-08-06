/* cgtgpu.cu - CUDA search engine: device field math, hashing and kernel.
 *
 * Shape of the search:
 *   Each lane owns an anchor point A = k*G. The lane walks outward in both
 *   directions, visiting A +/- i*G for i in [1, CGT_STRIDE_HALF]. Both signs
 *   share one x-coordinate chain, and negating a point costs nothing on the
 *   hash side beyond a parity flip -- so we get 2 candidate keys per EC step.
 *
 *   Point addition needs a modular inverse, which is the expensive part. We
 *   batch them: run the whole lane collecting denominators, then invert the
 *   running product once and unwind backwards (Montgomery's trick). That turns
 *   N inversions into 1 inversion + 3N multiplies.
 */

#include "cgtgpu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

/* ---------------- device-side 256-bit field arithmetic ---------------- */
/* Values are 4x64-bit little-endian limbs, held in registers as u64[4]. */

typedef unsigned long long u64;

#define CGT_DEV_PC 0x1000003D1ULL   /* p = 2^256 - 0x1000003D1 */

/* add with carry chain via PTX so the compiler keeps it in the carry flag */
__device__ __forceinline__ void devAdd(u64 *r, const u64 *a, const u64 *b) {
  asm volatile(
    "add.cc.u64  %0, %4, %8;  \n\t"
    "addc.cc.u64 %1, %5, %9;  \n\t"
    "addc.cc.u64 %2, %6, %10; \n\t"
    "addc.u64    %3, %7, %11; \n\t"
    : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3])
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
}

__device__ __forceinline__ u64 devAddCarry(u64 *r, const u64 *a, const u64 *b) {
  u64 c;
  asm volatile(
    "add.cc.u64  %0, %5, %9;  \n\t"
    "addc.cc.u64 %1, %6, %10; \n\t"
    "addc.cc.u64 %2, %7, %11; \n\t"
    "addc.cc.u64 %3, %8, %12; \n\t"
    "addc.u64    %4, 0, 0;    \n\t"
    : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]), "=l"(c)
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
  return c;
}

__device__ __forceinline__ u64 devSubBorrow(u64 *r, const u64 *a, const u64 *b) {
  u64 bw;
  asm volatile(
    "sub.cc.u64  %0, %5, %9;  \n\t"
    "subc.cc.u64 %1, %6, %10; \n\t"
    "subc.cc.u64 %2, %7, %11; \n\t"
    "subc.cc.u64 %3, %8, %12; \n\t"
    "subc.u64    %4, 0, 0;    \n\t"
    : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]), "=l"(bw)
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));
  return bw;   /* 0xFFFF... if a < b, else 0 */
}

__device__ __constant__ u64 CGT_DEV_P[4] = {
  0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
  0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

__device__ __forceinline__ void devSet(u64 *r, const u64 *a) {
  r[0]=a[0]; r[1]=a[1]; r[2]=a[2]; r[3]=a[3];
}

__device__ __forceinline__ bool devIsZero(const u64 *a) {
  return (a[0] | a[1] | a[2] | a[3]) == 0ULL;
}

__device__ __forceinline__ bool devEq(const u64 *a, const u64 *b) {
  return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
}

/* subtract p once if the value is >= p */
__device__ __forceinline__ void devTrim(u64 *a) {
  u64 t[4];
  if (devSubBorrow(t, a, CGT_DEV_P) == 0) devSet(a, t);
}

__device__ __forceinline__ void devFAdd(u64 *r, const u64 *a, const u64 *b) {
  u64 c = devAddCarry(r, a, b);
  if (c) {
    /* 2^256 == PC (mod p) */
    u64 t[4], one[4] = { CGT_DEV_PC, 0, 0, 0 };
    devAdd(t, r, one);
    devSet(r, t);
  }
  devTrim(r);
}

__device__ __forceinline__ void devFSub(u64 *r, const u64 *a, const u64 *b) {
  u64 bw = devSubBorrow(r, a, b);
  if (bw) {
    u64 t[4];
    devAdd(t, r, CGT_DEV_P);
    devSet(r, t);
  }
}

__device__ __forceinline__ void devFNeg(u64 *r, const u64 *a) {
  if (devIsZero(a)) { r[0]=r[1]=r[2]=r[3]=0; return; }
  devSubBorrow(r, CGT_DEV_P, a);
}

/* 256x256 -> 512.
 *
 * Written straight in PTX because the hardware carry flag is the whole point
 * here. Expressed in C the accumulation needs `(s < pl)` comparisons to
 * recover each carry, which is three extra instructions per limb product and
 * serialises on the comparison; `mad.lo.cc.u64` / `madc.hi.cc.u64` do the same
 * work with the carry riding in the flag for free.
 *
 * Each row after the first runs in two passes: all four low halves down one
 * carry chain, then all four high halves down a second one shifted up a limb.
 * Interleaving them instead would need two live carry flags, and there is only
 * one. Overflow past w7 cannot happen - the product is below 2^512 by
 * construction - so the last chain drops its carry out.
 */
__device__ __forceinline__ void devMulWide(u64 *lo, u64 *hi,
                                           const u64 *a, const u64 *b) {
  u64 w0, w1, w2, w3, w4, w5, w6, w7;
  asm(
    /* ---- row 0: a0 * b, straight into w0..w4 ---- */
    "mul.lo.u64     %0, %8, %12;      \n\t"
    "mul.hi.u64     %1, %8, %12;      \n\t"
    "mad.lo.cc.u64  %1, %8, %13, %1;  \n\t"
    "madc.hi.u64    %2, %8, %13,  0;  \n\t"
    "mad.lo.cc.u64  %2, %8, %14, %2;  \n\t"
    "madc.hi.u64    %3, %8, %14,  0;  \n\t"
    "mad.lo.cc.u64  %3, %8, %15, %3;  \n\t"
    "madc.hi.u64    %4, %8, %15,  0;  \n\t"
    "mov.u64        %5, 0;            \n\t"
    "mov.u64        %6, 0;            \n\t"
    "mov.u64        %7, 0;            \n\t"
    /* ---- row 1: a1 * b into w1..w5 ---- */
    "mad.lo.cc.u64  %1, %9, %12, %1;  \n\t"
    "madc.lo.cc.u64 %2, %9, %13, %2;  \n\t"
    "madc.lo.cc.u64 %3, %9, %14, %3;  \n\t"
    "madc.lo.cc.u64 %4, %9, %15, %4;  \n\t"
    "addc.u64       %5, %5, 0;        \n\t"
    "mad.hi.cc.u64  %2, %9, %12, %2;  \n\t"
    "madc.hi.cc.u64 %3, %9, %13, %3;  \n\t"
    "madc.hi.cc.u64 %4, %9, %14, %4;  \n\t"
    "madc.hi.cc.u64 %5, %9, %15, %5;  \n\t"
    "addc.u64       %6, %6, 0;        \n\t"
    /* ---- row 2: a2 * b into w2..w6 ---- */
    "mad.lo.cc.u64  %2, %10, %12, %2; \n\t"
    "madc.lo.cc.u64 %3, %10, %13, %3; \n\t"
    "madc.lo.cc.u64 %4, %10, %14, %4; \n\t"
    "madc.lo.cc.u64 %5, %10, %15, %5; \n\t"
    "addc.u64       %6, %6, 0;        \n\t"
    "mad.hi.cc.u64  %3, %10, %12, %3; \n\t"
    "madc.hi.cc.u64 %4, %10, %13, %4; \n\t"
    "madc.hi.cc.u64 %5, %10, %14, %5; \n\t"
    "madc.hi.cc.u64 %6, %10, %15, %6; \n\t"
    "addc.u64       %7, %7, 0;        \n\t"
    /* ---- row 3: a3 * b into w3..w7 ---- */
    "mad.lo.cc.u64  %3, %11, %12, %3; \n\t"
    "madc.lo.cc.u64 %4, %11, %13, %4; \n\t"
    "madc.lo.cc.u64 %5, %11, %14, %5; \n\t"
    "madc.lo.cc.u64 %6, %11, %15, %6; \n\t"
    "addc.u64       %7, %7, 0;        \n\t"
    "mad.hi.cc.u64  %4, %11, %12, %4; \n\t"
    "madc.hi.cc.u64 %5, %11, %13, %5; \n\t"
    "madc.hi.cc.u64 %6, %11, %14, %6; \n\t"
    "madc.hi.u64    %7, %11, %15, %7; \n\t"
    : "=l"(w0), "=l"(w1), "=l"(w2), "=l"(w3),
      "=l"(w4), "=l"(w5), "=l"(w6), "=l"(w7)
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
      "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3]));

  lo[0] = w0; lo[1] = w1; lo[2] = w2; lo[3] = w3;
  hi[0] = w4; hi[1] = w5; hi[2] = w6; hi[3] = w7;
}

/* fold a 512-bit product down mod p using 2^256 == PC
 *
 * PC is 33 bits, so hi*PC lands in 289 bits: four limbs plus a small fifth.
 * Folding that fifth limb back in costs one 64x64 product whose high half is
 * at most 2 bits, hence the (tl, th, 0, 0) shape of the second addition.
 * Afterwards the value is below 2^256, and 2^256 - p == PC, so a single
 * conditional subtract of p finishes the job.
 */
__device__ __forceinline__ void devFReduce(u64 *r, const u64 *lo, const u64 *hi) {
  u64 a0, a1, a2, a3, c2;
  asm(
    /* acc = lo + hi * PC, low halves then high halves */
    "{                                       \n\t"
    ".reg .u64 t4, tl, th;                   \n\t"
    "mad.lo.cc.u64  %0, %5, %13, %9;         \n\t"
    "madc.lo.cc.u64 %1, %6, %13, %10;        \n\t"
    "madc.lo.cc.u64 %2, %7, %13, %11;        \n\t"
    "madc.lo.cc.u64 %3, %8, %13, %12;        \n\t"
    "addc.u64       t4, 0, 0;                \n\t"
    "mad.hi.cc.u64  %1, %5, %13, %1;         \n\t"
    "madc.hi.cc.u64 %2, %6, %13, %2;         \n\t"
    "madc.hi.cc.u64 %3, %7, %13, %3;         \n\t"
    "madc.hi.u64    t4, %8, %13, t4;         \n\t"
    /* fold the surviving top limb back in */
    "mul.lo.u64     tl, t4, %13;             \n\t"
    "mul.hi.u64     th, t4, %13;             \n\t"
    "add.cc.u64     %0, %0, tl;              \n\t"
    "addc.cc.u64    %1, %1, th;              \n\t"
    "addc.cc.u64    %2, %2, 0;               \n\t"
    "addc.cc.u64    %3, %3, 0;               \n\t"
    "addc.u64       %4, 0, 0;                \n\t"
    "}                                       \n\t"
    : "=l"(a0), "=l"(a1), "=l"(a2), "=l"(a3), "=l"(c2)
    : "l"(hi[0]), "l"(hi[1]), "l"(hi[2]), "l"(hi[3]),
      "l"(lo[0]), "l"(lo[1]), "l"(lo[2]), "l"(lo[3]),
      "l"(CGT_DEV_PC));

  /* One more wrap is possible; PC is tiny so this cannot cascade further. */
  u64 back = c2 * CGT_DEV_PC;
  asm("add.cc.u64  %0, %0, %4; \n\t"
      "addc.cc.u64 %1, %1, 0;  \n\t"
      "addc.cc.u64 %2, %2, 0;  \n\t"
      "addc.u64    %3, %3, 0;  \n\t"
      : "+l"(a0), "+l"(a1), "+l"(a2), "+l"(a3) : "l"(back));

  r[0] = a0; r[1] = a1; r[2] = a2; r[3] = a3;
  devTrim(r);
}

__device__ __forceinline__ void devFMul(u64 *r, const u64 *a, const u64 *b) {
  u64 lo[4], hi[4];
  devMulWide(lo, hi, a, b);
  devFReduce(r, lo, hi);
}

/* Squaring, done as its own routine rather than devFMul(a,a).
 *
 * Of the sixteen limb products in a general multiply, six appear twice (a_i*a_j
 * and a_j*a_i) and only four are distinct diagonals. Summing the six once,
 * doubling the whole thing, then adding the diagonals gets the same 512-bit
 * result from ten products instead of sixteen. The inverse alone runs 255
 * squarings, so this is the hottest single routine in the kernel.
 */
__device__ __forceinline__ void devSqrWide(u64 *lo, u64 *hi, const u64 *a) {
  u64 w0, w1, w2, w3, w4, w5, w6, w7;
  asm(
    /* ---- off-diagonal half: a0a1, a0a2, a0a3, a1a2, a1a3, a2a3 ---- */
    "mul.lo.u64     %1, %8, %9;       \n\t"   /* a0*a1 -> w1,w2 */
    "mul.hi.u64     %2, %8, %9;       \n\t"
    "mad.lo.cc.u64  %2, %8, %10, %2;  \n\t"   /* a0*a2 -> w2,w3 */
    "madc.hi.u64    %3, %8, %10,  0;  \n\t"
    "mad.lo.cc.u64  %3, %8, %11, %3;  \n\t"   /* a0*a3 -> w3,w4 */
    "madc.hi.u64    %4, %8, %11,  0;  \n\t"
    "mad.lo.cc.u64  %3, %9, %10, %3;  \n\t"   /* a1*a2 -> w3,w4 */
    "madc.hi.cc.u64 %4, %9, %10, %4;  \n\t"
    "addc.u64       %5, 0, 0;         \n\t"
    "mad.lo.cc.u64  %4, %9, %11, %4;  \n\t"   /* a1*a3 -> w4,w5 */
    "madc.hi.cc.u64 %5, %9, %11, %5;  \n\t"
    "addc.u64       %6, 0, 0;         \n\t"
    "mad.lo.cc.u64  %5, %10, %11, %5; \n\t"   /* a2*a3 -> w5,w6 */
    "madc.hi.u64    %6, %10, %11, %6; \n\t"
    /* ---- double it: every cross term is counted twice.
       The cross sum reaches only as high as w6, so the single bit shifted out
       of w6 is the whole of w7 - grab it before the chain overwrites w6. ---- */
    "shr.u64        %7, %6, 63;       \n\t"
    "add.cc.u64     %1, %1, %1;       \n\t"
    "addc.cc.u64    %2, %2, %2;       \n\t"
    "addc.cc.u64    %3, %3, %3;       \n\t"
    "addc.cc.u64    %4, %4, %4;       \n\t"
    "addc.cc.u64    %5, %5, %5;       \n\t"
    "addc.u64       %6, %6, %6;       \n\t"
    /* ---- diagonals: a_i^2 at limbs 2i, 2i+1 ---- */
    "mul.lo.u64     %0, %8, %8;       \n\t"
    "mad.hi.cc.u64  %1, %8, %8, %1;   \n\t"
    "madc.lo.cc.u64 %2, %9, %9, %2;   \n\t"
    "madc.hi.cc.u64 %3, %9, %9, %3;   \n\t"
    "madc.lo.cc.u64 %4, %10, %10, %4; \n\t"
    "madc.hi.cc.u64 %5, %10, %10, %5; \n\t"
    "madc.lo.cc.u64 %6, %11, %11, %6; \n\t"
    "madc.hi.u64    %7, %11, %11, %7; \n\t"
    : "=l"(w0), "=l"(w1), "=l"(w2), "=l"(w3),
      "=l"(w4), "=l"(w5), "=l"(w6), "=l"(w7)
    : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]));

  lo[0] = w0; lo[1] = w1; lo[2] = w2; lo[3] = w3;
  hi[0] = w4; hi[1] = w5; hi[2] = w6; hi[3] = w7;
}

__device__ __forceinline__ void devFSqr(u64 *r, const u64 *a) {
  u64 lo[4], hi[4];
  devSqrWide(lo, hi, a);
  devFReduce(r, lo, hi);
}

/* repeated squaring: r = a^(2^n) */
__device__ __forceinline__ void devFSqrN(u64 *r, const u64 *a, int n) {
  devSet(r, a);
  for (int i = 0; i < n; i++) devFSqr(r, r);
}

/* Modular inverse, used once per batch thanks to Montgomery's trick.
 *
 * Fermat says a^(p-2), but evaluating that bit-by-bit costs 256 squarings plus
 * a multiply for every set bit - and p-2 is almost solid ones, so ~500 field
 * multiplies. secp256k1's prime has enough structure to do far better: build
 * runs of ones by squaring a shorter run and folding it back in, which reaches
 * the same exponent in 255 squarings and 15 multiplies.
 *
 * Naming below follows the run length: xN holds a^(2^N - 1).
 */
__device__ void devFInv(u64 *r, const u64 *a) {
  u64 x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4], x88[4];
  u64 x176[4], x220[4], x223[4], t[4];

  devFSqr(x2, a);        devFMul(x2, x2, a);      /* a^3      */
  devFSqr(x3, x2);       devFMul(x3, x3, a);      /* a^7      */

  devFSqrN(x6,   x3,   3);  devFMul(x6,   x6,   x3);
  devFSqrN(x9,   x6,   3);  devFMul(x9,   x9,   x3);
  devFSqrN(x11,  x9,   2);  devFMul(x11,  x11,  x2);
  devFSqrN(x22,  x11, 11);  devFMul(x22,  x22,  x11);
  devFSqrN(x44,  x22, 22);  devFMul(x44,  x44,  x22);
  devFSqrN(x88,  x44, 44);  devFMul(x88,  x88,  x44);
  devFSqrN(x176, x88, 88);  devFMul(x176, x176, x88);
  devFSqrN(x220, x176,44);  devFMul(x220, x220, x44);
  devFSqrN(x223, x220, 3);  devFMul(x223, x223, x3);

  /* tail: the low bits of p-2 are 0x...FFFFFC2D, spelled out here */
  devFSqrN(t, x223, 23);  devFMul(t, t, x22);
  devFSqrN(t, t,     5);  devFMul(t, t, a);
  devFSqrN(t, t,     3);  devFMul(t, t, x2);
  devFSqrN(t, t,     2);  devFMul(r, t, a);
}


/* ================= device SHA-256 ================= */

__device__ __constant__ unsigned int CGT_DK[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

typedef unsigned int u32;

__device__ __forceinline__ u32 devRotR(u32 x, int n) {
  return (x >> n) | (x << (32 - n));
}

#define DS0(x) (devRotR(x,2)  ^ devRotR(x,13) ^ devRotR(x,22))
#define DS1(x) (devRotR(x,6)  ^ devRotR(x,11) ^ devRotR(x,25))
#define DG0(x) (devRotR(x,7)  ^ devRotR(x,18) ^ ((x) >> 3))
#define DG1(x) (devRotR(x,17) ^ devRotR(x,19) ^ ((x) >> 10))
#define DCH(x,y,z)  (((x) & (y)) ^ (~(x) & (z)))
#define DMAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))

/* One SHA-256 compression over a 16-word block.
   The message schedule is kept as a rolling 16-word window rather than a
   w[64] array: w[i] only ever reads back 2/7/15/16 slots, so the window is
   sufficient and saves 48 registers per thread. At this kernel's register
   pressure that difference is what decides occupancy. */
__device__ __forceinline__ void devShaBlock(u32 *s, u32 *w) {
  u32 a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];

  #pragma unroll
  for (int i = 0; i < 64; i++) {
    u32 wi;
    if (i < 16) {
      wi = w[i];
    } else {
      /* w[i] = G1(w[i-2]) + w[i-7] + G0(w[i-15]) + w[i-16], all mod 16 */
      wi = DG1(w[(i-2) & 15]) + w[(i-7) & 15] + DG0(w[(i-15) & 15]) + w[i & 15];
      w[i & 15] = wi;
    }
    u32 t1 = h + DS1(e) + DCH(e,f,g) + CGT_DK[i] + wi;
    u32 t2 = DS0(a) + DMAJ(a,b,c);
    h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
  }
  s[0]+=a; s[1]+=b; s[2]+=c; s[3]+=d; s[4]+=e; s[5]+=f; s[6]+=g; s[7]+=h;
}

/* SHA-256 of a 33-byte compressed pubkey. One block: 33 bytes + pad + length. */
__device__ void devSha33(const u32 xw[8], u32 prefix, u32 out[8]) {
  u32 w[16];
  /* byte layout: [prefix][x0..x31], big-endian words */
  w[0] = (prefix << 24) | (xw[0] >> 8);
  #pragma unroll
  for (int i = 1; i < 8; i++) w[i] = (xw[i-1] << 24) | (xw[i] >> 8);
  w[8] = (xw[7] << 24) | 0x00800000u;    /* last x byte then the 0x80 pad */
  #pragma unroll
  for (int i = 9; i < 15; i++) w[i] = 0;
  w[15] = 33 * 8;                        /* message length in bits */

  out[0]=0x6a09e667; out[1]=0xbb67ae85; out[2]=0x3c6ef372; out[3]=0xa54ff53a;
  out[4]=0x510e527f; out[5]=0x9b05688c; out[6]=0x1f83d9ab; out[7]=0x5be0cd19;
  devShaBlock(out, w);
}

/* SHA-256 of a 65-byte uncompressed pubkey. Two blocks. */
__device__ void devSha65(const u32 xw[8], const u32 yw[8], u32 out[8]) {
  u32 w[16];
  /* block 1: 0x04 || X(32) || Y(0..30) -> 64 bytes */
  w[0] = (0x04u << 24) | (xw[0] >> 8);
  #pragma unroll
  for (int i = 1; i < 8; i++) w[i] = (xw[i-1] << 24) | (xw[i] >> 8);
  w[8] = (xw[7] << 24) | (yw[0] >> 8);
  #pragma unroll
  for (int i = 9; i < 16; i++) w[i] = (yw[i-9] << 24) | (yw[i-8] >> 8);

  out[0]=0x6a09e667; out[1]=0xbb67ae85; out[2]=0x3c6ef372; out[3]=0xa54ff53a;
  out[4]=0x510e527f; out[5]=0x9b05688c; out[6]=0x1f83d9ab; out[7]=0x5be0cd19;
  devShaBlock(out, w);

  /* block 2: last byte of Y, pad, length */
  w[0] = (yw[7] << 24) | 0x00800000u;
  #pragma unroll
  for (int i = 1; i < 15; i++) w[i] = 0;
  w[15] = 65 * 8;
  devShaBlock(out, w);
}

/* ================= device RIPEMD-160 ================= */
/* The round list lives in cgtrmd.cuh, generated by genrmd.ps1 so that every
   message index and rotate amount is a compile-time literal. Table lookups
   would push the 16-word message block into local memory. */

#include "cgtrmd.cuh"



/* full hash160 for a compressed pubkey */
__device__ __forceinline__ void devH160Comp(const u32 xw[8], u32 prefix, u32 out[5]) {
  u32 sh[8];
  devSha33(xw, prefix, sh);
  cgtRmd160(sh, out);
}

/* full hash160 for an uncompressed pubkey */
__device__ __forceinline__ void devH160Uncomp(const u32 xw[8], const u32 yw[8], u32 out[5]) {
  u32 sh[8];
  devSha65(xw, yw, sh);
  cgtRmd160(sh, out);
}

/* ================= on-device bloom screen ================= */
/* Mirrors the host filter exactly (see cgtpool.cpp cgtBloomHashes): the digest
   words are read out directly, the fifth folded into h2, then walked as
   h1 + i*h2 masked to the table size. Any change here must change the host
   side too, or the GPU will screen out real hits. */

__device__ __forceinline__ void devBloomHashes(const u32 h[5], u64 &a, u64 &b) {
  a = (u64)h[0] | ((u64)h[1] << 32);
  b = (u64)h[2] | ((u64)h[3] << 32);
  b ^= (u64)h[4] * 0x9E3779B97F4A7C15ULL;
  b |= 1ULL;
}

__device__ __forceinline__ bool devBloomTest(const unsigned char *bloom,
                                             u64 mask, int khash,
                                             const u32 h[5]) {
  u64 a, b;
  devBloomHashes(h, a, b);
  #pragma unroll 1
  for (int i = 0; i < khash; i++) {
    u64 bit = (a + (u64)i * b) & mask;
    if (!(bloom[bit >> 3] & (unsigned char)(1u << (bit & 7)))) return false;
  }
  return true;
}

/* ================= on-device x-coordinate screen (pubkey engine) =========== */
/* The pubkey engine never hashes. A candidate point's x-coordinate IS the
   comparison key, so the bloom is built over x directly and a surviving
   candidate is confirmed against the uploaded target list on the spot.

   Mixing: x is a field element, already uniform over 256 bits, so - exactly as
   with the RIPEMD digest on the address side - there is nothing for a hash
   function to improve. The limbs are folded pairwise and one multiply ties the
   halves together so h2 is not a bare copy of the top half.

   Must stay bit-identical to cgtXBloomHashes in cgtpkpool.cpp. */

/* First-stage screen, in shared memory.
   Dropping SHA-256 + RIPEMD-160 only bought 17% (3813 -> 4453 Mkey/s), which
   says the hashing was never the wall: the bloom was. Eight probes at
   unpredictable offsets into a multi-megabyte array are eight L2/DRAM round
   trips per key, and no amount of arithmetic saved offsets that.

   So screen first out of a table small enough to live in shared memory. One
   probe, one bank access, and the global bloom is only consulted by what gets
   through. At 64 Kbit a run against a handful of targets passes this stage
   about once every 65536 keys, so the global array is effectively never read.

   Sizing: 8 KiB of the SM's shared memory, which the kernel is not otherwise
   using - the per-lane inversion scratch is thread-local, not shared. */
#define CGTPK_PRE_BITS  65536
#define CGTPK_PRE_WORDS (CGTPK_PRE_BITS / 64)

/* Bit index for x. Host and device must agree; cgtPreIndexHost mirrors this. */
__device__ __forceinline__ u64 devPreIndex(const u64 *x) {
  return (x[0] ^ x[1]) & (u64)(CGTPK_PRE_BITS - 1);
}

__device__ __forceinline__ bool devPreTest(const u64 *pre, const u64 *x) {
  u64 i = devPreIndex(x);
  return ((pre[i >> 6] >> (i & 63)) & 1ULL) != 0ULL;
}

__device__ __forceinline__ void devXBloomHashes(const u64 *x, u64 &a, u64 &b) {
  a = x[0] ^ x[1];
  b = x[2] ^ x[3];
  b ^= a * 0x9E3779B97F4A7C15ULL;
  b |= 1ULL;                     /* keep the stride odd */
}

__device__ __forceinline__ bool devXBloomTest(const unsigned char *bloom,
                                              u64 mask, int khash,
                                              const u64 *x) {
  u64 a, b;
  devXBloomHashes(x, a, b);
  #pragma unroll 1
  for (int i = 0; i < khash; i++) {
    u64 bit = (a + (u64)i * b) & mask;
    if (!(bloom[bit >> 3] & (unsigned char)(1u << (bit & 7)))) return false;
  }
  return true;
}

/* Exact match against the uploaded x list. Only ever reached on a bloom hit,
   which for a real search is a handful of times in the whole run, so a linear
   scan is the right shape - it costs no setup and no extra resident memory. */
__device__ __forceinline__ bool devXMatch(const u64 *targets, int count,
                                          const u64 *x) {
  #pragma unroll 1
  for (int i = 0; i < count; i++)
    if (devEq(x, &targets[i * 4])) return true;
  return false;
}

/* record a pubkey candidate: the x-coordinate goes back verbatim, since the
   host confirms by recomputing the point rather than by re-hashing */
__device__ __forceinline__ void devEmitX(cgt_hit *hits, unsigned int *count,
                                         u32 lane, u32 step, u32 flags,
                                         const u64 *x) {
  unsigned int slot = atomicAdd(count, 1u);
  if (slot >= CGT_HIT_SLOTS) return;
  hits[slot].lane  = lane;
  hits[slot].step  = step;
  hits[slot].flags = flags;
  /* h160[] is reused as a 160-bit window onto x purely so the hit record has
     one layout. The host does not read it - it rebuilds the key from lane+step
     and verifies against the target list - but carrying it makes a hit
     self-describing when debugging. */
  #pragma unroll
  for (int i = 0; i < 5; i++)
    hits[slot].h160[i] = (u32)(x[i >> 1] >> ((i & 1) ? 32 : 0));
}

/* record a candidate for the host to confirm */
__device__ __forceinline__ void devEmit(cgt_hit *hits, unsigned int *count,
                                        u32 lane, u32 step, u32 flags,
                                        const u32 h[5]) {
  unsigned int slot = atomicAdd(count, 1u);
  if (slot >= CGT_HIT_SLOTS) return;      /* overflow: host warns and rescans */
  hits[slot].lane  = lane;
  hits[slot].step  = step;
  hits[slot].flags = flags;
  #pragma unroll
  for (int i = 0; i < 5; i++) hits[slot].h160[i] = h[i];
}

/* pack a field element into 8 big-endian u32 words for hashing */
__device__ __forceinline__ void devPackBE(const u64 *v, u32 w[8]) {
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    u64 limb = v[3 - i];
    w[2*i]     = (u32)(limb >> 32);
    w[2*i + 1] = (u32)(limb);
  }
}

/* small kernel: slide each lane's anchor forward by the grid stride */
__global__ void cgtAdvanceKernel(u64 *anchorX, u64 *anchorY,
                                 const u64 *gridX, const u64 *gridY) {
  int lane = blockIdx.x * blockDim.x + threadIdx.x;
  u64 ax[4], ay[4], gx[4], gy[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    ax[i] = anchorX[lane * 4 + i];
    ay[i] = anchorY[lane * 4 + i];
    gx[i] = gridX[i];
    gy[i] = gridY[i];
  }
  /* A' = A + G_grid via the same point-addition the kernel uses */
  u64 dx[4], num[4], dInv[4], lam[4], xr[4], yr[4], t[4];
  devFSub(dx, gx, ax);
  devFSub(num, gy, ay);
  devFInv(dInv, dx);
  devFMul(lam, num, dInv);
  devFSqr(xr, lam); devFSub(xr, xr, ax); devFSub(xr, xr, gx);
  devFSub(t, ax, xr); devFMul(yr, lam, t); devFSub(yr, yr, ay);
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    anchorX[lane * 4 + i] = xr[i];
    anchorY[lane * 4 + i] = yr[i];
  }
}

/* seed kernel for random mode: anchor[i] = base + laneOff[i].
   Lane 0's offset is the point at infinity (cannot be represented in affine),
   so it gets the base verbatim. Every other lane does one point addition. */
__global__ void cgtSeedKernel(u64 *anchorX, u64 *anchorY,
                              const u64 *baseX, const u64 *baseY,
                              const u64 *laneOffX, const u64 *laneOffY) {
  int lane = blockIdx.x * blockDim.x + threadIdx.x;
  u64 bx[4], by[4], ox[4], oy[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    bx[i] = baseX[i];
    by[i] = baseY[i];
    ox[i] = laneOffX[lane * 4 + i];
    oy[i] = laneOffY[lane * 4 + i];
  }
  if (lane == 0) {
    /* lane 0: offset is zero (infinity), result is the base */
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      anchorX[i] = bx[i];
      anchorY[i] = by[i];
    }
  } else {
    /* anchor = base + offset */
    u64 dx[4], num[4], dInv[4], lam[4], xr[4], yr[4], t[4];
    devFSub(dx, ox, bx);
    devFSub(num, oy, by);
    devFInv(dInv, dx);
    devFMul(lam, num, dInv);
    devFSqr(xr, lam); devFSub(xr, xr, bx); devFSub(xr, xr, ox);
    devFSub(t, bx, xr); devFMul(yr, lam, t); devFSub(yr, yr, by);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      anchorX[lane * 4 + i] = xr[i];
      anchorY[lane * 4 + i] = yr[i];
    }
  }
}

/* ================= main kernel ================= */
/* Each lane walks +/- i*G around its anchor A for i in [1, CGT_STRIDE_HALF].
   P = A + i*G and Q = A - i*G share the denominator (x_step - x_anchor), so
   one modular inverse serves both. We batch the inversions: collect all
   denominators, invert their product once (Montgomery's trick), then unwind. */

/* The two output flavours are template parameters rather than arguments so the
   unused one compiles away entirely. It matters more than it looks: the
   uncompressed digest hashes 65 bytes, which is a second SHA-256 block, and
   leaving that branch in costs registers in every thread even when nobody
   takes it. The default search is compressed-only. */
/* The read-only inputs are marked const __restrict__ to state that they never
   alias the hits/hitCount output. Measured effect on this compiler is nil
   (+0.09%, inside run-to-run noise) - nvcc already infers it here - but the
   annotation is correct and costs nothing. */
template <bool COMP, bool UNCOMP>
__global__ void cgtKernel(
  const u64 * __restrict__ anchorX, const u64 * __restrict__ anchorY,
  const u64 * __restrict__ stepX, const u64 * __restrict__ stepY,
  const u64 * __restrict__ jumpX, const u64 * __restrict__ jumpY,
  const unsigned char * __restrict__ bloom, u64 bloomMask, int bloomHashes,
  cgt_hit * __restrict__ hits, unsigned int * __restrict__ hitCount
) {
  int lane = blockIdx.x * blockDim.x + threadIdx.x;

  u64 ax[4], ay[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    ax[i] = anchorX[lane * 4 + i];
    ay[i] = anchorY[lane * 4 + i];
  }

  /* The lane sweeps CGT_ROUNDS consecutive windows without returning to the
     host. Each round hashes [A - HALF, A + HALF], then slides A forward by
     CGT_LANE_KEYS using the jump point J = CGT_LANE_KEYS * G. The jump's own
     denominator is appended to the same batch, so the slide costs one extra
     slot in an inversion we were paying for anyway. */
  #pragma unroll 1
  for (int round = 0; round < CGT_ROUNDS; round++) {

  u32 base = (u32)round * (u32)CGT_LANE_KEYS;

  /* The walk covers anchor +/- i for i in [1, CGT_STRIDE_HALF], so the anchor
     key itself would fall through the gap. Hash it here: we already hold its
     coordinates, so it costs no field work at all. Step code CGT_STEP_ANCHOR
     tells the host the delta is zero. */
  {
    u32 xw[8], yw[8], h[5];
    devPackBE(ax, xw);
    devPackBE(ay, yw);
    if (COMP) {
      u32 pref = (yw[7] & 1u) ? 0x03u : 0x02u;
      devH160Comp(xw, pref, h);
      if (devBloomTest(bloom, bloomMask, bloomHashes, h))
        devEmit(hits, hitCount, (u32)lane, base + CGT_STEP_ANCHOR_OFF, 0, h);
    }
    if (UNCOMP) {
      devH160Uncomp(xw, yw, h);
      if (devBloomTest(bloom, bloomMask, bloomHashes, h))
        devEmit(hits, hitCount, (u32)lane, base + CGT_STEP_ANCHOR_OFF, 2, h);
    }
  }

  /* Forward pass: prefix[i] = d_0 * ... * d_(i-1), so prefix[0] is 1. We only
     store the prefixes; each d_i is one subtraction to recompute. Slot
     CGT_STRIDE_HALF is the jump denominator (jx - ax), so the single inversion
     below covers the whole window plus the anchor slide.

     Splitting this running product across several independent accumulators to
     shorten the dependency chain was measured and is slower: each extra
     accumulator costs ~16 registers, and at 128 registers x 512 threads the SM
     register file is already full, so the only way to fit is a cap that spills
     or a smaller block. Both cost more than the overlap wins (3570 and 3047 vs
     3867 Mkey/s). One accumulator is the right shape here. */
  u64 prefix[CGT_STRIDE_HALF + 1][4];
  u64 run[4] = {1, 0, 0, 0};

  #pragma unroll 1
  for (int i = 0; i < CGT_STRIDE_HALF; i++) {
    devSet(prefix[i], run);
    u64 sx[4], dx[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) sx[j] = stepX[i * 4 + j];
    devFSub(dx, sx, ax);           /* d_i = x_step - x_anchor */
    devFMul(run, run, dx);
  }
  {
    devSet(prefix[CGT_STRIDE_HALF], run);
    u64 dx[4];
    devFSub(dx, jumpX, ax);        /* jump denominator */
    devFMul(run, run, dx);
  }

  /* One inversion for the whole batch. */
  u64 acc[4];
  devFInv(acc, run);               /* acc = 1 / (d_0 * ... * d_N) */

  /* Slide the anchor first, while its inverse is still the outermost term.
     newA = A + J, computed but not applied until the walk below is done. */
  u64 nax[4], nay[4];
  {
    u64 dInv[4];
    devFMul(dInv, acc, prefix[CGT_STRIDE_HALF]);
    u64 num[4], lam[4], t[4];
    devFSub(num, jumpY, ay);
    devFMul(lam, num, dInv);
    devFSqr(nax, lam);
    devFSub(nax, nax, ax);
    devFSub(nax, nax, jumpX);
    devFSub(t, ax, nax);
    devFMul(nay, lam, t);
    devFSub(nay, nay, ay);

    u64 dx[4];
    devFSub(dx, jumpX, ax);
    devFMul(acc, acc, dx);         /* fold it out again */
  }

  /* Backward pass: 1/d_i = acc * prefix[i], then fold d_i into acc so the next
     (earlier) slot sees the inverse of the product up to but excluding
     itself. */
  #pragma unroll 1
  for (int i = CGT_STRIDE_HALF - 1; i >= 0; i--) {
    u64 dInv[4];
    devFMul(dInv, acc, prefix[i]);

    u64 sx[4], sy[4], num_plus[4], num_minus[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      sx[j] = stepX[i * 4 + j];
      sy[j] = stepY[i * 4 + j];
    }

    /* P = A + i*G: num = sy - ay */
    devFSub(num_plus, sy, ay);
    /* Q = A - i*G: num = -sy - ay. Reached as (p - sy) - ay rather than
       negating the sum, which skips the add's carry fixup and its trim. */
    u64 negsy[4];
    devFNeg(negsy, sy);
    devFSub(num_minus, negsy, ay);

    /* Compute P = A + i*G */
    u64 lam[4], xr[4], yr[4], t[4];
    devFMul(lam, num_plus, dInv);
    devFSqr(xr, lam);
    devFSub(xr, xr, ax);
    devFSub(xr, xr, sx);
    devFSub(t, ax, xr);
    devFMul(yr, lam, t);
    devFSub(yr, yr, ay);

    u32 xw[8], yw[8], h[5];
    devPackBE(xr, xw);
    devPackBE(yr, yw);

    if (COMP) {
      u32 pref = (yw[7] & 1u) ? 0x03u : 0x02u;
      devH160Comp(xw, pref, h);
      if (devBloomTest(bloom, bloomMask, bloomHashes, h))
        devEmit(hits, hitCount, (u32)lane, base + (u32)(i * 2), 0, h);
    }
    if (UNCOMP) {
      devH160Uncomp(xw, yw, h);
      if (devBloomTest(bloom, bloomMask, bloomHashes, h))
        devEmit(hits, hitCount, (u32)lane, base + (u32)(i * 2), 2, h);
    }

    /* Compute Q = A - i*G. It reuses the SAME denominator inverse, but the
       numerator differs, so lambda differs and xr must be recomputed:
       lam_Q = -(sy+ay)/(sx-ax) is not +/- lam_P, hence lam_Q^2 != lam_P^2. */
    devFMul(lam, num_minus, dInv);
    devFSqr(xr, lam);
    devFSub(xr, xr, ax);
    devFSub(xr, xr, sx);
    devFSub(t, ax, xr);
    devFMul(yr, lam, t);
    devFSub(yr, yr, ay);

    devPackBE(xr, xw);
    devPackBE(yr, yw);

    if (COMP) {
      u32 pref = (yw[7] & 1u) ? 0x03u : 0x02u;
      devH160Comp(xw, pref, h);
      if (devBloomTest(bloom, bloomMask, bloomHashes, h))
        devEmit(hits, hitCount, (u32)lane, base + (u32)(i * 2 + 1), 1, h);
    }
    if (UNCOMP) {
      devH160Uncomp(xw, yw, h);
      if (devBloomTest(bloom, bloomMask, bloomHashes, h))
        devEmit(hits, hitCount, (u32)lane, base + (u32)(i * 2 + 1), 3, h);
    }

    /* Fold d_i back into the accumulator. */
    u64 dx[4];
    devFSub(dx, sx, ax);
    devFMul(acc, acc, dx);
  }

  /* Advance to the next window. */
  devSet(ax, nax);
  devSet(ay, nay);

  } /* rounds */
}

/* ================= pubkey kernel ================= */
/* Identical walk structure to the address kernel, but at the point where
   coordinates are computed we compare x directly against the target list
   instead of hashing. SHA-256 + RIPEMD-160 is ~51% of the address-search
   runtime (measured 7928 vs 3869 Mkey/s on this hardware), so skipping it
   should reach 7000-8000 Mkey/s — exactly the user's expectation for pubkey
   mode. The kernel reuses the same step tables, anchor seeds, and grid stride
   as the address engine; only the bloom/target buffers differ. */

template <bool COMP, bool UNCOMP>
__global__ void cgtPubkeyKernel(
  const u64 * __restrict__ anchorX, const u64 * __restrict__ anchorY,
  const u64 * __restrict__ stepX, const u64 * __restrict__ stepY,
  const u64 * __restrict__ jumpX, const u64 * __restrict__ jumpY,
  const unsigned char * __restrict__ bloom, u64 bloomMask, int bloomHashes,
  const u64 * __restrict__ pre,
  const u64 * __restrict__ targets, int targetCount,
  cgt_hit * __restrict__ hits, unsigned int * __restrict__ hitCount
) {
  int lane = blockIdx.x * blockDim.x + threadIdx.x;

  /* Stage-one table, pulled in once per launch and then read from shared for
     every one of the ~4.7 M keys this block goes on to check. */
  __shared__ u64 sPre[CGTPK_PRE_WORDS];
  for (int i = threadIdx.x; i < CGTPK_PRE_WORDS; i += blockDim.x)
    sPre[i] = pre[i];
  __syncthreads();

  u64 ax[4], ay[4];
  #pragma unroll
  for (int i = 0; i < 4; i++) {
    ax[i] = anchorX[lane * 4 + i];
    ay[i] = anchorY[lane * 4 + i];
  }

  #pragma unroll 1
  for (int round = 0; round < CGT_ROUNDS; round++) {

  u32 base = (u32)round * (u32)CGT_LANE_KEYS;

  /* Check the anchor itself. */
  if (COMP || UNCOMP) {
    if (devPreTest(sPre, ax) &&
        devXBloomTest(bloom, bloomMask, bloomHashes, ax)) {
      if (devXMatch(targets, targetCount, ax)) {
        u32 f = COMP ? 0 : 2;
        devEmitX(hits, hitCount, (u32)lane, base + CGT_STEP_ANCHOR_OFF, f, ax);
      }
    }
  }

  /* Forward pass: same batch-inversion structure. */
  u64 prefix[CGT_STRIDE_HALF + 1][4];
  u64 run[4] = {1, 0, 0, 0};

  #pragma unroll 1
  for (int i = 0; i < CGT_STRIDE_HALF; i++) {
    devSet(prefix[i], run);
    u64 sx[4], dx[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) sx[j] = stepX[i * 4 + j];
    devFSub(dx, sx, ax);
    devFMul(run, run, dx);
  }
  {
    devSet(prefix[CGT_STRIDE_HALF], run);
    u64 dx[4];
    devFSub(dx, jumpX, ax);
    devFMul(run, run, dx);
  }

  u64 acc[4];
  devFInv(acc, run);

  /* Slide anchor. */
  u64 nax[4], nay[4];
  {
    u64 dInv[4];
    devFMul(dInv, acc, prefix[CGT_STRIDE_HALF]);
    u64 num[4], lam[4], t[4];
    devFSub(num, jumpY, ay);
    devFMul(lam, num, dInv);
    devFSqr(nax, lam);
    devFSub(nax, nax, ax);
    devFSub(nax, nax, jumpX);
    devFSub(t, ax, nax);
    devFMul(nay, lam, t);
    devFSub(nay, nay, ay);

    u64 dx[4];
    devFSub(dx, jumpX, ax);
    devFMul(acc, acc, dx);
  }

  /* Backward pass: P = A + i*G and Q = A - i*G. */
  #pragma unroll 1
  for (int i = CGT_STRIDE_HALF - 1; i >= 0; i--) {
    u64 dInv[4];
    devFMul(dInv, acc, prefix[i]);

    u64 sx[4], sy[4], num[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      sx[j] = stepX[i * 4 + j];
      sy[j] = stepY[i * 4 + j];
    }

    /* This is where the pubkey engine actually pulls ahead, and it is not the
       hashing. x(P) = lam^2 - ax - sx involves no y of the result, so the two
       multiplies the address kernel spends forming y for each point - it needs
       y's parity for the compressed prefix, and the full y uncompressed - are
       simply never issued. Nine field multiplies per pair of keys become seven. */
    u64 lam[4], xr[4];

    /* P = A + i*G */
    devFSub(num, sy, ay);
    devFMul(lam, num, dInv);
    devFSqr(xr, lam);
    devFSub(xr, xr, ax);
    devFSub(xr, xr, sx);

    /* Compare x directly: shared-memory screen, then the global bloom, then the
       exact list. Almost every key stops at the first test.
       One test covers both serializations - compressed and uncompressed forms of
       a point share an x - so the host takes the format from the target record. */
    if (COMP || UNCOMP) {
      if (devPreTest(sPre, xr) &&
          devXBloomTest(bloom, bloomMask, bloomHashes, xr)) {
        if (devXMatch(targets, targetCount, xr)) {
          u32 f = COMP ? 0 : 2;
          devEmitX(hits, hitCount, (u32)lane, base + (u32)(i * 2), f, xr);
        }
      }
    }

    /* Q = A - i*G. Its slope is -(sy + ay)/d, and only lam^2 is used from here
       on, so the negation is dropped and (sy + ay) goes in directly. */
    devFAdd(num, sy, ay);
    devFMul(lam, num, dInv);
    devFSqr(xr, lam);
    devFSub(xr, xr, ax);
    devFSub(xr, xr, sx);

    if (COMP || UNCOMP) {
      if (devPreTest(sPre, xr) &&
          devXBloomTest(bloom, bloomMask, bloomHashes, xr)) {
        if (devXMatch(targets, targetCount, xr)) {
          u32 f = COMP ? 1 : 3;
          devEmitX(hits, hitCount, (u32)lane, base + (u32)(i * 2 + 1), f, xr);
        }
      }
    }

    u64 dx[4];
    devFSub(dx, sx, ax);
    devFMul(acc, acc, dx);
  }

  devSet(ax, nax);
  devSet(ay, nay);

  } /* rounds */
}

/* ================= host C++ wrapper ================= */

static int devCoresPerSM(int major, int minor) {
  int sm = (major << 4) | minor;
  if (sm >= 0x120) return 192;   /* sm_120 Blackwell  */
  if (sm >= 0x100) return 128;   /* sm_100 Hopper     */
  if (sm >= 0x90)  return 128;   /* sm_90  Hopper     */
  if (sm >= 0x80)  return 64;    /* sm_80  Ampere     */
  if (sm >= 0x75)  return 64;    /* sm_75  Turing     */
  if (sm >= 0x70)  return 64;    /* sm_70  Volta      */
  if (sm >= 0x60)  return 64;    /* sm_60  Pascal     */
  return 64;
}

bool cgtGpuList(std::vector<cgt_devinfo> &out) {
  out.clear();
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return false;
  for (int i = 0; i < n; i++) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
    cgt_devinfo d;
    d.id = i;
    d.name = p.name;
    d.major = p.major;
    d.minor = p.minor;
    d.smCount = p.multiProcessorCount;
    d.coresPerSM = devCoresPerSM(p.major, p.minor);
    d.memBytes = p.totalGlobalMem;
    out.push_back(d);
  }
  return !out.empty();
}

CgtGpu::CgtGpu()
  : lanes(0), blocks(0), threadsPerBlock(CGT_BLOCK_THREADS),
    wantComp(true), wantUncomp(false), ready(false),
    dAnchorX(NULL), dAnchorY(NULL), dBloom(NULL),
    dTargets(NULL), dPre(NULL), targetCount(0), pkMode(false),
    dHits(NULL), dHitCount(NULL), dStepX(NULL), dStepY(NULL),
    dJumpX(NULL), dJumpY(NULL), dGridX(NULL), dGridY(NULL),
    dLaneOffX(NULL), dLaneOffY(NULL), dBaseX(NULL), dBaseY(NULL),
    bloomBits(0), bloomHashes(0) {}

CgtGpu::~CgtGpu() {
  if (dAnchorX) cudaFree(dAnchorX);
  if (dAnchorY) cudaFree(dAnchorY);
  if (dBloom)   cudaFree(dBloom);
  if (dTargets) cudaFree(dTargets);
  if (dPre)     cudaFree(dPre);
  if (dHits)    cudaFree(dHits);
  if (dHitCount) cudaFree(dHitCount);
  if (dStepX)   cudaFree(dStepX);
  if (dStepY)   cudaFree(dStepY);
  if (dJumpX)   cudaFree(dJumpX);
  if (dJumpY)   cudaFree(dJumpY);
  if (dGridX)   cudaFree(dGridX);
  if (dGridY)   cudaFree(dGridY);
  if (dLaneOffX) cudaFree(dLaneOffX);
  if (dLaneOffY) cudaFree(dLaneOffY);
  if (dBaseX)   cudaFree(dBaseX);
  if (dBaseY)   cudaFree(dBaseY);
}

bool CgtGpu::open(int id, std::string &err) {
  if (cudaSetDevice(id) != cudaSuccess) {
    err = "cudaSetDevice failed";
    return false;
  }
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, id) != cudaSuccess) {
    err = "cudaGetDeviceProperties failed";
    return false;
  }
  dev.id = id;
  dev.name = p.name;
  dev.major = p.major;
  dev.minor = p.minor;
  dev.smCount = p.multiProcessorCount;
  dev.coresPerSM = devCoresPerSM(p.major, p.minor);
  dev.memBytes = p.totalGlobalMem;

  /* grid: blocks * threads = total lanes */
  blocks = dev.smCount * CGT_BLOCKS_PER_SM;
  lanes = blocks * threadsPerBlock;

  /* Shrink the grid if the card cannot back its local memory.
     Each lane keeps a private prefix-product array of CGT_STRIDE_HALF+1 field
     elements - about 32 KiB at the default window - and the driver reserves
     that for every resident thread out of ordinary VRAM. The default geometry
     wants roughly a gigabyte, which is nothing on a large card and more than a
     4 GB card has spare. Ask the compiler what the frame actually costs rather
     than guessing, then cut the grid until it fits. Fewer lanes only means
     fewer keys in flight; the search stays correct. */
  {
    cudaFuncAttributes fa;
    size_t frame = 0;
    if (cudaFuncGetAttributes(&fa, cgtKernel<true, false>) == cudaSuccess)
      frame = fa.localSizeBytes;
    if (cudaFuncGetAttributes(&fa, cgtKernel<false, true>) == cudaSuccess &&
        fa.localSizeBytes > frame) frame = fa.localSizeBytes;
    if (cudaFuncGetAttributes(&fa, cgtKernel<true, true>) == cudaSuccess &&
        fa.localSizeBytes > frame) frame = fa.localSizeBytes;

    size_t freeB = 0, totalB = 0;
    if (frame > 0 && cudaMemGetInfo(&freeB, &totalB) == cudaSuccess) {
      /* Leave headroom: the tables, bloom filter and hit buffer also live here,
         and a display-attached GPU has a desktop to keep alive. */
      size_t budget = (size_t)((double)freeB * 0.70);
      size_t maxLanes = budget / frame;
      if (maxLanes < 32) maxLanes = 32;

      if ((size_t)lanes > maxLanes) {
        /* Prefer trimming block size - it keeps every SM occupied - and only
           drop whole blocks once a block is already down to one warp. */
        while ((size_t)lanes > maxLanes && threadsPerBlock > 32) {
          threadsPerBlock /= 2;
          lanes = blocks * threadsPerBlock;
        }
        while ((size_t)lanes > maxLanes && blocks > dev.smCount) {
          blocks /= 2;
          lanes = blocks * threadsPerBlock;
        }
        while ((size_t)lanes > maxLanes && blocks > 1) {
          blocks /= 2;
          lanes = blocks * threadsPerBlock;
        }
        printf("[+] Grid reduced to %d x %d lanes to fit %.0f MB of GPU memory\n",
               blocks, threadsPerBlock, (double)freeB / (1024.0 * 1024.0));
      }
    }
  }

  /* allocate device buffers */
  size_t sz = (size_t)lanes * 4 * sizeof(uint64_t);
  if (cudaMalloc(&dAnchorX, sz) != cudaSuccess ||
      cudaMalloc(&dAnchorY, sz) != cudaSuccess) {
    err = "cudaMalloc anchor buffers failed";
    return false;
  }

  /* Precompute the step table i*G for i in [1, CGT_STRIDE_HALF], uploaded
     once. Built incrementally: acc <- acc + G, so the whole table costs
     CGT_STRIDE_HALF point additions rather than a scalar multiply each. */
  cgtBuildLadder();
  const cgt_pt &G = cgtGenerator();
  std::vector<uint64_t> stepXh(CGT_STRIDE_HALF * 4);
  std::vector<uint64_t> stepYh(CGT_STRIDE_HALF * 4);
  cgt_pt acc = G;                       /* i = 1 */
  for (int i = 0; i < CGT_STRIDE_HALF; i++) {
    for (int j = 0; j < 4; j++) {
      stepXh[i * 4 + j] = acc.x.w[j];
      stepYh[i * 4 + j] = acc.y.w[j];
    }
    cgt_pt nxt;
    cgtPtAdd(nxt, acc, G);
    acc = nxt;
  }
  sz = CGT_STRIDE_HALF * 4 * sizeof(uint64_t);
  if (cudaMalloc(&dStepX, sz) != cudaSuccess ||
      cudaMalloc(&dStepY, sz) != cudaSuccess ||
      cudaMemcpy(dStepX, &stepXh[0], sz, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dStepY, &stepYh[0], sz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMalloc/Memcpy step table failed";
    return false;
  }

  /* Jump point J = CGT_LANE_KEYS * G: how far a lane slides between rounds.
     `acc` already holds (CGT_STRIDE_HALF + 1) * G from the loop above, and
     CGT_LANE_KEYS = 2*CGT_STRIDE_HALF + 1, so add CGT_STRIDE_HALF more. */
  for (int i = 0; i < CGT_STRIDE_HALF; i++) {
    cgt_pt nxt;
    cgtPtAdd(nxt, acc, G);
    acc = nxt;
  }
  uint64_t jx[4], jy[4];
  for (int j = 0; j < 4; j++) { jx[j] = acc.x.w[j]; jy[j] = acc.y.w[j]; }
  sz = 4 * sizeof(uint64_t);
  if (cudaMalloc(&dJumpX, sz) != cudaSuccess ||
      cudaMalloc(&dJumpY, sz) != cudaSuccess ||
      cudaMemcpy(dJumpX, jx, sz, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dJumpY, jy, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMalloc/Memcpy jump point failed";
    return false;
  }

  /* Grid stride G_grid = (lanes * CGT_LANE_REGION) * G: how far the whole grid
     advances per pass. That scalar is far too large to reach by repeated
     addition, so compute it directly with one scalar multiply. */
  cgt_u256 gridScalar;
  cgtSetU64(gridScalar, (uint64_t)lanes * (uint64_t)CGT_LANE_REGION);
  cgt_pt gridPt;
  cgtPtMulG(gridPt, gridScalar);
  uint64_t gx[4], gy[4];
  for (int j = 0; j < 4; j++) { gx[j] = gridPt.x.w[j]; gy[j] = gridPt.y.w[j]; }
  if (cudaMalloc(&dGridX, sz) != cudaSuccess ||
      cudaMalloc(&dGridY, sz) != cudaSuccess ||
      cudaMemcpy(dGridX, gx, sz, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dGridY, gy, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMalloc/Memcpy grid stride failed";
    return false;
  }

  /* Per-lane offsets L_i = (i * CGT_LANE_REGION) * G, built once.
     Random mode needs a fresh anchor for every lane on every pass. Deriving
     them from scalars costs one scalar multiply per lane, which at this lane
     count dominated the search completely - the GPU sat idle while the host
     ran 17920 ladders. With this table a pass needs one scalar multiply for
     the random base and a kernel adds the offsets.
     Built incrementally by repeated addition of R = CGT_LANE_REGION * G, so
     the whole table is `lanes` point additions rather than `lanes` ladders. */
  cgt_u256 regionScalar;
  cgtSetU64(regionScalar, (uint64_t)CGT_LANE_REGION);
  cgt_pt regionPt;
  cgtPtMulG(regionPt, regionScalar);

  std::vector<uint64_t> offXh((size_t)lanes * 4);
  std::vector<uint64_t> offYh((size_t)lanes * 4);
  cgt_pt off = regionPt;                 /* i = 1; lane 0 is handled separately */
  for (int i = 1; i < lanes; i++) {
    for (int j = 0; j < 4; j++) {
      offXh[(size_t)i * 4 + j] = off.x.w[j];
      offYh[(size_t)i * 4 + j] = off.y.w[j];
    }
    cgt_pt nxt;
    cgtPtAdd(nxt, off, regionPt);
    off = nxt;
  }
  /* Lane 0's offset is the point at infinity, which affine addition cannot
     represent. Left as zero; the seed kernel special-cases it. */
  for (int j = 0; j < 4; j++) { offXh[j] = 0; offYh[j] = 0; }

  sz = (size_t)lanes * 4 * sizeof(uint64_t);
  if (cudaMalloc(&dLaneOffX, sz) != cudaSuccess ||
      cudaMalloc(&dLaneOffY, sz) != cudaSuccess ||
      cudaMemcpy(dLaneOffX, &offXh[0], sz, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dLaneOffY, &offYh[0], sz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMalloc/Memcpy lane offset table failed";
    return false;
  }
  if (cudaMalloc(&dBaseX, 4 * sizeof(uint64_t)) != cudaSuccess ||
      cudaMalloc(&dBaseY, 4 * sizeof(uint64_t)) != cudaSuccess) {
    err = "cudaMalloc base point failed";
    return false;
  }

  if (cudaMalloc(&dHits, CGT_HIT_SLOTS * sizeof(cgt_hit)) != cudaSuccess ||
      cudaMalloc(&dHitCount, sizeof(uint32_t)) != cudaSuccess) {
    err = "cudaMalloc hit buffers failed";
    return false;
  }

  ready = true;
  return true;
}

bool CgtGpu::uploadFilter(const uint8_t *bloom, size_t bloomBytes,
                          uint64_t bits, int hashes, std::string &err) {
  if (!ready) { err = "device not open"; return false; }
  if (dBloom) cudaFree(dBloom);
  if (dTargets) { cudaFree(dTargets); dTargets = NULL; }
  if (dPre)     { cudaFree(dPre);     dPre     = NULL; }
  targetCount = 0;
  pkMode = false;
  if (cudaMalloc(&dBloom, bloomBytes) != cudaSuccess ||
      cudaMemcpy(dBloom, bloom, bloomBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMalloc/Memcpy bloom failed";
    dBloom = NULL;
    return false;
  }
  /* The host sizes the table to a power of two, so the kernel can index with a
     mask instead of a 64-bit remainder. Refuse anything else rather than
     silently screening out real hits. */
  if (bits == 0 || (bits & (bits - 1)) != 0) {
    err = "bloom size is not a power of two";
    return false;
  }
  bloomBits = bits;
  bloomHashes = hashes;
  return true;
}

bool CgtGpu::uploadPubkeyFilter(const uint8_t *bloom, size_t bloomBytes,
                                uint64_t bits, int hashes,
                                const uint64_t *xCoords, size_t targetCount,
                                std::string &err) {
  if (!ready) { err = "device not open"; return false; }
  if (dBloom) cudaFree(dBloom);
  if (dTargets) cudaFree(dTargets);
  if (dPre) cudaFree(dPre);
  dBloom = dTargets = dPre = NULL;
  pkMode = true;
  this->targetCount = (int)targetCount;

  if (bits == 0 || (bits & (bits - 1)) != 0) {
    err = "bloom size is not a power of two";
    return false;
  }

  if (cudaMalloc(&dBloom, bloomBytes) != cudaSuccess ||
      cudaMemcpy(dBloom, bloom, bloomBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "bloom upload failed";
    return false;
  }

  size_t tsz = targetCount * 4 * sizeof(uint64_t);
  if (cudaMalloc(&dTargets, tsz) != cudaSuccess ||
      cudaMemcpy(dTargets, xCoords, tsz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "x-coordinate upload failed";
    return false;
  }

  /* Build the shared-memory stage-one table here rather than in the pool: it is
     a property of this kernel's geometry, and its index function has to track
     devPreIndex exactly, which lives in this file. */
  {
    std::vector<uint64_t> pre(CGTPK_PRE_WORDS, 0);
    for (size_t i = 0; i < targetCount; i++) {
      const uint64_t *x = &xCoords[i * 4];
      uint64_t idx = (x[0] ^ x[1]) & (uint64_t)(CGTPK_PRE_BITS - 1);
      pre[idx >> 6] |= 1ULL << (idx & 63);
    }
    size_t psz = CGTPK_PRE_WORDS * sizeof(uint64_t);
    if (cudaMalloc(&dPre, psz) != cudaSuccess ||
        cudaMemcpy(dPre, &pre[0], psz, cudaMemcpyHostToDevice) != cudaSuccess) {
      err = "pre-screen upload failed";
      return false;
    }
  }

  bloomBits = bits;
  bloomHashes = hashes;
  return true;
}

void CgtGpu::setModes(bool compressed, bool uncompressed) {
  wantComp = compressed;
  wantUncomp = uncompressed;
}

bool CgtGpu::seedAnchors(const std::vector<cgt_u256> &keys, std::string &err) {
  if (!ready) { err = "device not open"; return false; }
  if ((int)keys.size() != lanes) {
    err = "key count mismatch";
    return false;
  }
  std::vector<uint64_t> xh(lanes * 4), yh(lanes * 4);
  for (int i = 0; i < lanes; i++) {
    cgt_pt p;
    cgtPtMulG(p, keys[i]);
    for (int j = 0; j < 4; j++) {
      xh[i * 4 + j] = p.x.w[j];
      yh[i * 4 + j] = p.y.w[j];
    }
  }
  size_t sz = (size_t)lanes * 4 * sizeof(uint64_t);
  if (cudaMemcpy(dAnchorX, &xh[0], sz, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dAnchorY, &yh[0], sz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMemcpy anchor failed";
    return false;
  }
  return true;
}

/* Random mode: derive every lane's anchor point on the device from a single
   host scalar multiply.

   The old path called cgtPtMulG once per lane, so a pass paid `lanes` ladders
   (17920 of them at the default geometry) before the kernel could start. That
   left the GPU idle for the great majority of every pass and pinned random
   mode near 40 Mkey/s while sequential mode ran two orders of magnitude
   faster. Lane i's anchor scalar is base + i*CGT_LANE_REGION, so its point is
   base*G + L_i where L_i comes from the table built in open(). One ladder for
   the base, then one point addition per lane on the device. */
bool CgtGpu::seedAnchorsRandom(const cgt_u256 &base, std::string &err) {
  if (!ready) { err = "device not open"; return false; }

  cgt_pt bp;
  cgtPtMulG(bp, base);
  uint64_t bx[4], by[4];
  for (int j = 0; j < 4; j++) { bx[j] = bp.x.w[j]; by[j] = bp.y.w[j]; }
  size_t sz = 4 * sizeof(uint64_t);
  if (cudaMemcpy(dBaseX, bx, sz, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(dBaseY, by, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMemcpy base point failed";
    return false;
  }

  cgtSeedKernel<<<blocks, threadsPerBlock>>>(
    (u64 *)dAnchorX, (u64 *)dAnchorY,
    (const u64 *)dBaseX, (const u64 *)dBaseY,
    (const u64 *)dLaneOffX, (const u64 *)dLaneOffY);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    err = std::string("seed kernel launch failed: ") + cudaGetErrorString(e);
    return false;
  }
  return true;
}

/* Slide every anchor forward by one whole grid stride, on device. This is what
   replaces re-deriving 17920 points from scalars on the host each pass: the
   host cost was dominating the search by more than an order of magnitude. */
bool CgtGpu::advanceAnchors(std::string &err) {
  if (!ready) { err = "device not open"; return false; }
  cgtAdvanceKernel<<<blocks, threadsPerBlock>>>(
    (uint64_t *)dAnchorX, (uint64_t *)dAnchorY,
    (const uint64_t *)dGridX, (const uint64_t *)dGridY);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    err = std::string("anchor advance launch failed: ") + cudaGetErrorString(e);
    return false;
  }
  return true;
}

bool CgtGpu::runPass(std::vector<cgt_hit> &hits, std::string &err) {
  if (!ready) { err = "device not open"; return false; }
  if (!dBloom) { err = "bloom not uploaded"; return false; }

  uint32_t zero = 0;
  if (cudaMemcpy(dHitCount, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
    err = "cudaMemcpy hitCount clear failed";
    return false;
  }

  /* Pick the kernel based on mode: pubkey-direct or hash-based address search. */
  if (pkMode) {
    /* Pubkey engine: x-coordinate comparison, no hashing. */
    if (!dTargets || !dPre) { err = "pubkey targets not uploaded"; return false; }
    #define CGTPK_LAUNCH(C, U)                                          \
      cgtPubkeyKernel<C, U><<<blocks, threadsPerBlock>>>(               \
        (const uint64_t *)dAnchorX, (const uint64_t *)dAnchorY,         \
        (const uint64_t *)dStepX, (const uint64_t *)dStepY,             \
        (const uint64_t *)dJumpX, (const uint64_t *)dJumpY,             \
        (const unsigned char *)dBloom, bloomBits - 1ULL, bloomHashes,   \
        (const uint64_t *)dPre,                                         \
        (const uint64_t *)dTargets, targetCount,                        \
        (cgt_hit *)dHits, (unsigned int *)dHitCount)

    if (wantComp && wantUncomp)       { CGTPK_LAUNCH(true, true);   }
    else if (wantComp)                { CGTPK_LAUNCH(true, false);  }
    else if (wantUncomp)              { CGTPK_LAUNCH(false, true);  }
    else                              { CGTPK_LAUNCH(true, false);  }

    #undef CGTPK_LAUNCH
  } else {
    /* Address engine: hash160 comparison. */
    #define CGT_LAUNCH(C, U)                                            \
      cgtKernel<C, U><<<blocks, threadsPerBlock>>>(                     \
        (const uint64_t *)dAnchorX, (const uint64_t *)dAnchorY,         \
        (const uint64_t *)dStepX, (const uint64_t *)dStepY,             \
        (const uint64_t *)dJumpX, (const uint64_t *)dJumpY,             \
        (const unsigned char *)dBloom, bloomBits - 1ULL, bloomHashes,   \
        (cgt_hit *)dHits, (unsigned int *)dHitCount)

    if (wantComp && wantUncomp)     { CGT_LAUNCH(true, true);   }
    else if (wantUncomp)            { CGT_LAUNCH(false, true);  }
    else                            { CGT_LAUNCH(true, false);  }

    #undef CGT_LAUNCH
  }

  if (cudaDeviceSynchronize() != cudaSuccess) {
    err = "kernel launch or sync failed";
    return false;
  }

  uint32_t count;
  if (cudaMemcpy(&count, dHitCount, sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
    err = "cudaMemcpy hitCount readback failed";
    return false;
  }

  if (count > CGT_HIT_SLOTS) {
    err = "hit buffer overflow; reduce stride or target count";
    count = CGT_HIT_SLOTS;
  }

  if (count > 0) {
    std::vector<cgt_hit> buf(count);
    if (cudaMemcpy(&buf[0], dHits, count * sizeof(cgt_hit), cudaMemcpyDeviceToHost) != cudaSuccess) {
      err = "cudaMemcpy hits readback failed";
      return false;
    }
    hits.insert(hits.end(), buf.begin(), buf.end());
  }

  return true;
}

uint64_t CgtGpu::keysPerPass(void) const {
  return (uint64_t)lanes * CGT_LANE_REGION;
}

/* Decode a step code back into the signed distance from the lane's starting
   anchor. The lane slid forward by CGT_LANE_KEYS per round, so:
     delta = round*CGT_LANE_KEYS + (local == ANCHOR ? 0 : +/-(local/2 + 1))
   which can be negative overall only when round == 0. */
void CgtGpu::offsetForStep(uint32_t step, cgt_u256 &delta, bool &negative) {
  uint32_t round = step / (uint32_t)CGT_LANE_KEYS;
  uint32_t local = step % (uint32_t)CGT_LANE_KEYS;

  int64_t forward = (int64_t)round * (int64_t)CGT_LANE_KEYS;
  if (local != CGT_STEP_ANCHOR_OFF) {
    int64_t i = (int64_t)(local / 2) + 1;
    forward += (local & 1u) ? -i : i;
  }

  if (forward < 0) { negative = true;  cgtSetU64(delta, (uint64_t)(-forward)); }
  else             { negative = false; cgtSetU64(delta, (uint64_t)forward); }
}
