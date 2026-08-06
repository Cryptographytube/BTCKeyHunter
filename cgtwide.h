/* cgtwide.h - portable 64x64->128 multiply and add/sub with carry.
   MSVC has no __uint128_t, so route through the x64 intrinsics there. */
#ifndef CGTWIDE_H
#define CGTWIDE_H

#include <stdint.h>

#if defined(_MSC_VER)
  #include <intrin.h>
  #pragma intrinsic(_umul128)

  /* lo = a*b low 64, hi = a*b high 64 */
  #define CGT_MUL64(lo, hi, a, b)  ((lo) = _umul128((a), (b), &(hi)))

  /* r = a + b + cin ; cout = carry (0/1) */
  static inline unsigned char cgt_addc(uint64_t a, uint64_t b,
                                       unsigned char cin, uint64_t *r) {
    return _addcarry_u64(cin, a, b, (unsigned long long *)r);
  }
  /* r = a - b - bin ; bout = borrow (0/1) */
  static inline unsigned char cgt_subb(uint64_t a, uint64_t b,
                                       unsigned char bin, uint64_t *r) {
    return _subborrow_u64(bin, a, b, (unsigned long long *)r);
  }
#else
  typedef unsigned __int128 cgt_uint128;

  #define CGT_MUL64(lo, hi, a, b)                       \
    do {                                                \
      cgt_uint128 _t = (cgt_uint128)(a) * (b);          \
      (lo) = (uint64_t)_t;                              \
      (hi) = (uint64_t)(_t >> 64);                      \
    } while (0)

  static inline unsigned char cgt_addc(uint64_t a, uint64_t b,
                                       unsigned char cin, uint64_t *r) {
    cgt_uint128 s = (cgt_uint128)a + b + cin;
    *r = (uint64_t)s;
    return (unsigned char)(s >> 64);
  }
  static inline unsigned char cgt_subb(uint64_t a, uint64_t b,
                                       unsigned char bin, uint64_t *r) {
    cgt_uint128 d = (cgt_uint128)a - b - bin;
    *r = (uint64_t)d;
    return (unsigned char)((d >> 64) & 1);
  }
#endif

#endif /* CGTWIDE_H */
