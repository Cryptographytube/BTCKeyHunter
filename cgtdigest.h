/* cgtdigest.h - SHA-256, RIPEMD-160, Base58Check address encoding. */
#ifndef CGTDIGEST_H
#define CGTDIGEST_H

#include "cgtdef.h"
#include <vector>

/* ---------------- SHA-256 ---------------- */
struct cgt_sha256_ctx {
  uint32_t s[8];
  uint8_t  buf[64];
  uint64_t bitlen;
  size_t   bufused;
};

void cgtSha256Init(cgt_sha256_ctx &ctx);
void cgtSha256Update(cgt_sha256_ctx &ctx, const uint8_t *data, size_t len);
void cgtSha256Final(cgt_sha256_ctx &ctx, uint8_t out[32]);
/* one-shot helper */
void cgtSha256(const uint8_t *data, size_t len, uint8_t out[32]);

/* ---------------- RIPEMD-160 ---------------- */
struct cgt_ripemd160_ctx {
  uint32_t s[5];
  uint8_t  buf[64];
  uint64_t bitlen;
  size_t   bufused;
};

void cgtRipemd160Init(cgt_ripemd160_ctx &ctx);
void cgtRipemd160Update(cgt_ripemd160_ctx &ctx, const uint8_t *data, size_t len);
void cgtRipemd160Final(cgt_ripemd160_ctx &ctx, uint8_t out[20]);
void cgtRipemd160(const uint8_t *data, size_t len, uint8_t out[20]);

/* ---------------- Hash160 (SHA-256 then RIPEMD-160) ---------------- */
void cgtHash160(const uint8_t *data, size_t len, uint8_t out[20]);

/* ---------------- Base58Check encoding ---------------- */
std::string cgtBase58CheckEncode(uint8_t version, const uint8_t *payload, size_t plen);
/* Decode a Base58Check string; returns true on success, fills version and payload. */
bool        cgtBase58CheckDecode(const std::string &s, uint8_t &version,
                                  std::vector<uint8_t> &payload);

/* P2PKH address from hash160 */
std::string cgtAddrFromHash160(const uint8_t h160[20]);

#endif /* CGTDIGEST_H */
