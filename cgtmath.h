/* cgtmath.h - host-side 256-bit arithmetic and secp256k1 group operations. */
#ifndef CGTMATH_H
#define CGTMATH_H

#include "cgtdef.h"

/* ---------------- plain 256-bit integer helpers ---------------- */
void      cgtSetZero(cgt_u256 &a);
void      cgtSetU64(cgt_u256 &a, uint64_t v);
bool      cgtIsZero(const cgt_u256 &a);
int       cgtCmp(const cgt_u256 &a, const cgt_u256 &b);
/* r = a + b, returns carry out */
uint64_t  cgtAdd(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
/* r = a - b, returns borrow out */
uint64_t  cgtSub(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
uint64_t  cgtAddU64(cgt_u256 &r, const cgt_u256 &a, uint64_t v);
void      cgtShiftRight1(cgt_u256 &a);
/* q = a / d, returns a % d. d must be non-zero. */
uint64_t  cgtDivU64(cgt_u256 &q, const cgt_u256 &a, uint64_t d);
/* r = a * v, truncated to 256 bits. */
void      cgtMulU64(cgt_u256 &r, const cgt_u256 &a, uint64_t v);
int       cgtBitLength(const cgt_u256 &a);
int       cgtGetBit(const cgt_u256 &a, int i);

/* text conversion; hex may carry an optional 0x prefix */
bool        cgtParseHex256(const std::string &txt, cgt_u256 &out);
std::string cgtToHex256(const cgt_u256 &a);      /* no leading zeros */
std::string cgtToHex256Pad(const cgt_u256 &a);   /* 64 nibbles       */
void        cgtToBytes32(const cgt_u256 &a, uint8_t out[32]);
void        cgtFromBytes32(cgt_u256 &a, const uint8_t in[32]);
/* approximate magnitude as a double, for probability/ETA math */
double      cgtToDouble(const cgt_u256 &a);

/* ---------------- secp256k1 field (mod p) ---------------- */
void cgtFieldAdd(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
void cgtFieldSub(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
void cgtFieldMul(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
void cgtFieldSqr(cgt_u256 &r, const cgt_u256 &a);
void cgtFieldInv(cgt_u256 &r, const cgt_u256 &a);
void cgtFieldNeg(cgt_u256 &r, const cgt_u256 &a);

/* scalar field (mod n) helpers used for key bookkeeping */
void cgtScalarAdd(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
void cgtScalarSub(cgt_u256 &r, const cgt_u256 &a, const cgt_u256 &b);
void cgtScalarNeg(cgt_u256 &r, const cgt_u256 &a);

/* curve order n and field prime p, exposed for range clamping */
const cgt_u256 &cgtCurveOrder(void);
const cgt_u256 &cgtFieldPrime(void);

/* ---------------- affine curve points ---------------- */
struct cgt_pt {
  cgt_u256 x;
  cgt_u256 y;
  bool     inf;
  cgt_pt() : inf(true) { cgtSetZero(x); cgtSetZero(y); }
};

void cgtPtDouble(cgt_pt &r, const cgt_pt &a);
void cgtPtAdd(cgt_pt &r, const cgt_pt &a, const cgt_pt &b);
void cgtPtNeg(cgt_pt &r, const cgt_pt &a);
bool cgtPtOnCurve(const cgt_pt &a);
/* r = k * G, plain double-and-add over the affine form */
void cgtPtMulG(cgt_pt &r, const cgt_u256 &k);
const cgt_pt &cgtGenerator(void);

/* Table of G * 2^i for i in [0,256), built once. Used to seed lane anchors. */
void cgtBuildLadder(void);
const cgt_pt *cgtLadder(void);

/* compressed / uncompressed SEC serialisation */
void cgtSerializePub(const cgt_pt &p, bool compressed, std::vector<uint8_t> &out);
/* parse 33-byte compressed or 65-byte uncompressed pubkey hex */
bool cgtParsePub(const std::string &hex, cgt_pt &out, bool *wasCompressed);

#endif /* CGTMATH_H */
