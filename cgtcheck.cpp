/* cgtcheck.cpp - standalone host self-test (built by `make selftest`). */
#include "cgtdef.h"
#include "cgtmath.h"
#include "cgtdigest.h"
#include "cgtpool.h"
#include "cgtspan.h"
#include <cstdio>
#include <cstring>

static int fails = 0;

static void ok(const char *name, bool cond) {
  printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name);
  if (!cond) fails++;
}

int main(void) {
  printf("CGTKEY host self-test\n");

  /* --- SHA-256 known vectors --- */
  uint8_t d[32];
  cgtSha256((const uint8_t *)"abc", 3, d);
  ok("sha256(\"abc\")",
     cgtBytesToHex(d, 32) ==
     "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD");

  cgtSha256((const uint8_t *)"", 0, d);
  ok("sha256(\"\")",
     cgtBytesToHex(d, 32) ==
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");

  /* --- RIPEMD-160 known vectors --- */
  uint8_t r[20];
  cgtRipemd160((const uint8_t *)"abc", 3, r);
  ok("ripemd160(\"abc\")",
     cgtBytesToHex(r, 20) == "8EB208F7E05D987A9B044A8E98C6B087F15A0BFC");

  cgtRipemd160((const uint8_t *)"", 0, r);
  ok("ripemd160(\"\")",
     cgtBytesToHex(r, 20) == "9C1185A5C5E9FC54612808977EE8F548B2258D31");

  /* --- secp256k1 generator sanity --- */
  ok("G on curve", cgtPtOnCurve(cgtGenerator()));

  cgt_pt p2;
  cgtPtDouble(p2, cgtGenerator());
  ok("2G on curve", cgtPtOnCurve(p2));
  ok("2G.x",
     cgtToHex256Pad(p2.x) ==
     "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");

  /* k*G via ladder must agree with repeated addition */
  cgt_u256 k;
  cgtSetU64(k, 1);
  cgt_pt g1;
  cgtPtMulG(g1, k);
  ok("1*G == G", cgtCmp(g1.x, cgtGenerator().x) == 0 &&
                 cgtCmp(g1.y, cgtGenerator().y) == 0);

  cgtSetU64(k, 2);
  cgt_pt gg2;
  cgtPtMulG(gg2, k);
  ok("2*G == double(G)", cgtCmp(gg2.x, p2.x) == 0 && cgtCmp(gg2.y, p2.y) == 0);

  /* --- field inversion round trip --- */
  cgt_u256 a, ai, prod, one;
  cgtParseHex256("A1B2C3D4E5F60718293A4B5C6D7E8F90"
                 "1122334455667788990AABBCCDDEEFF0", a);
  cgtFieldInv(ai, a);
  cgtFieldMul(prod, a, ai);
  cgtSetU64(one, 1);
  ok("a * a^-1 == 1", cgtCmp(prod, one) == 0);

  /* --- Bitcoin puzzle #1: privkey 1 -> known address --- */
  cgtSetU64(k, 1);
  cgt_pt pk;
  cgtPtMulG(pk, k);
  std::vector<uint8_t> ser;
  cgtSerializePub(pk, true, ser);
  uint8_t h[20];
  cgtHash160(&ser[0], ser.size(), h);
  ok("puzzle#1 addr (compressed)",
     cgtAddrFromHash160(h) == "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH");

  cgtSerializePub(pk, false, ser);
  cgtHash160(&ser[0], ser.size(), h);
  ok("privkey 1 addr (uncompressed)",
     cgtAddrFromHash160(h) == "1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm");

  /* --- Bitcoin puzzle keys. The puzzle addresses are COMPRESSED. --- */
  cgtParseHex256("D2C55", k);
  cgtPtMulG(pk, k);
  cgtSerializePub(pk, true, ser);
  cgtHash160(&ser[0], ser.size(), h);
  ok("puzzle#20 addr",
     cgtAddrFromHash160(h) == "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum");

  cgtSetU64(k, 3);
  cgtPtMulG(pk, k);
  cgtSerializePub(pk, true, ser);
  cgtHash160(&ser[0], ser.size(), h);
  ok("puzzle#2 addr",
     cgtAddrFromHash160(h) == "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb");

  cgtSetU64(k, 7);
  cgtPtMulG(pk, k);
  cgtSerializePub(pk, true, ser);
  cgtHash160(&ser[0], ser.size(), h);
  ok("puzzle#3 addr",
     cgtAddrFromHash160(h) == "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA");

  /* --- Base58Check round trip --- */
  uint8_t ver;
  std::vector<uint8_t> pl;
  ok("b58 decode puzzle#20",
     cgtBase58CheckDecode("1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum", ver, pl) &&
     ver == 0 && pl.size() == 20);

  /* --- range parsing --- */
  cgt_u256 s, e;
  ok("range \"66\"", cgtParseRange("66", s, e) &&
     cgtToHex256(s) == "20000000000000000" &&
     cgtToHex256(e) == "3FFFFFFFFFFFFFFFF");

  ok("range hex pair",
     cgtParseRange("67AE147AE147AE1484:68F5C28F5C28F5C297", s, e) &&
     cgtToHex256(s) == "67AE147AE147AE1484" &&
     cgtToHex256(e) == "68F5C28F5C28F5C297");

  /* --- pool matching --- */
  CgtPool pool;
  ok("pool add addr", pool.addAddress("1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum"));
  pool.finalize();
  cgtParseHex256("D2C55", k);
  cgtPtMulG(pk, k);
  cgtSerializePub(pk, true, ser);   // compressed, to match the address
  cgtHash160(&ser[0], ser.size(), h);
  ok("pool match hit", pool.match(h));
  h[0] ^= 0xFF;
  ok("pool match miss", !pool.match(h));

  printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASS",
         fails, fails == 1 ? "" : "s");
  return fails ? 1 : 0;
}
