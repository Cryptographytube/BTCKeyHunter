# CGTKEY — Bitcoin Private Key Recovery Tool

**Author:** Cryptographytube  
**Version:** 1.0

High-performance CUDA-based Bitcoin private key search engine with dual modes: hash-based address search and x-coordinate-only public key search.

---
## CMD
```
CGTbuild.bat
```
## COMPILE
```
[*] CGTbuild: STRIDE_HALF=2048  arch=sm_120  retries=12
[*] CUDA    : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
[*] VS      : C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools
[*] Toolset : 14.29.30133  (known-good for CUDA 13.1)
[*] Stage 1: host .cpp at /Od ...
[*] attempt 1/12: compile device obj ...
cgtcli.obj
cgtmath.obj
cgtdigest.obj
cgtpool.obj
cgtpkpool.obj
cgtspan.obj
cgtgpu.obj
    smoke test ...
    smoke OK

[+] BUILD OK: cgtkey.exe  (also copied to cgtkey_2048.exe)
[+]   arch=sm_120  toolset=14.29.30133  CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

```
## RUN

```
cgtkey_2048.exe -r 135 -i btc.txt -R

```
```
[+] CGTKEY v.1.0
[+] Search: 2 targets [P2PKH]
[+] Start Wed Aug 19 09:12:58 2026
[+] Random mode
[+] Range (2^135)
[+] from : 0x4000000000000000000000000000000000
[+] to   : 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
[+] GPU 0: NVIDIA GeForce RTX 5070 Ti (70 SM, cap 12.0, 16303 MB)
[+] Lanes 35840, 1.17 G keys per pass
[+] Coverage: 18539391133371680.00 P blocks of 1.17 G keys, each visited once in random order
[+] Engine: address (hash160)

[GPU 3820.46 Mkey/s][Total 2^39.15][Prob 2.8e-27%][50% in 9.032e+22y][Found 0][55657E067FAD471BA9F8EDAC48B7832BA7]

```

## Features

- **Dual Search Engines**
  - **Address Mode:** Hash160-based P2PKH address search (3,800 Mkey/s)
  - **Pubkey Mode:** Direct x-coordinate comparison — **2.64× faster** (10,000 Mkey/s)
  
- **Perfect Coverage**
  - Sequential mode: exhaustive coverage, every key checked exactly once
  - Random mode: permuted block order with full-range guarantee
  
- **GPU-Accelerated**
  - Batch elliptic curve operations with Montgomery's trick
  - Symmetric ±walk: P = A + i*G and Q = A − i*G share one inversion
  - Shared-memory + bloom filter pre-screening
  
- **Resume Support**
  - Checkpoint every 10 seconds in backup mode (`-b`)
  - Survive crashes, Ctrl+C, and console close events
  
- **Fat Binary**
  - Single executable runs on Maxwell through Blackwell (sm_50–sm_120+)

---

## Performance

| Mode | Speed | Hardware |
|---|---|---|
| Address (hash160) | 3,893 Mkey/s | RTX 5070 Ti |
| **Pubkey (x-compare)** | **10,113 Mkey/s** | RTX 5070 Ti |

Pubkey mode skips SHA-256 + RIPEMD-160 entirely and never computes y-coordinates.

---

## Installation

### Prerequisites
- **CUDA Toolkit** 11.0+ ([Download](https://developer.nvidia.com/cuda-downloads))
- **Visual Studio 2019+** (Windows) or **GCC 7+** (Linux)
- **NVIDIA GPU** with Compute Capability 5.0+

### Build

**Linux / macOS:**
```bash
cd cgtkey
make                  # Fat binary for all architectures
make ARCH=89          # Single architecture (faster rebuild)
```

**Windows:**
```powershell
cd cgtkey
.\build.ps1                    # Fat binary
.\build.ps1 -Arch 120          # Single architecture
.\build.ps1 -SelfTest          # Build + run host tests
```

---

## Usage

### Basic Syntax
```bash
cgtkey -r <bits> [-a <address> | -p <pubkey> | -i <file>] [options]
```

### Search Modes

**Address Search** (hash160):
```bash
cgtkey -a 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb -r 35
```

**Public Key Search** (x-coordinate, 2.64× faster):
```bash
# Compressed
cgtkey -p 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d -r 35

# Uncompressed
cgtkey -p 04f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d8fde318a70a6dbed63953f4bb93087820e5ccb071faa514204ad19650e5a86ee -r 35
```

**File List** (all addresses or all pubkeys):
```bash
cgtkey -i targets.txt -r 40
```

### Range Specification

**Bit range** (2^n to 2^(n+1)−1):
```bash
cgtkey -p <pubkey> -r 71
```

**Custom hex range**:
```bash
cgtkey -p <pubkey> -r 8000000000000:FFFFFFFFFFFFFF
```

### Options

| Flag | Description |
|---|---|
| `-R` | Random mode: permuted block order |
| `-b` | Backup mode: resume from last checkpoint |
| `-u` | Uncompressed targets (address mode only) |
| `-G <ID>` | GPU device ID (default: 0) |
| `-h` | Display help |

---

## Examples

### Sequential Puzzle 35
```bash
cgtkey -p 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d -r 35
```
**Output:**
```
KEY FOUND
Target  : 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d
Address : 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb
PrivKey : 0x00000000000000000000000000000000000000000000000000000004AED21170
Format  : compressed

Scanned 17.62 G keys in 1.8 s (9822.98 Mkey/s)
```

### Random Mode with Resume
```bash
cgtkey -p 03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4 -r 40 -R -b
```

### Mixed File (auto-fallback to address mode)
```bash
# targets.txt:
# 02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d
# 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb

cgtkey -i targets.txt -r 35
# Output: "File mixes addresses and public keys: using the address engine."
```

---

## Output Files

| File | Description |
|---|---|
| `cgtkey_found.txt` | Discovered private keys (append mode) |
| `cgtkey_resume.txt` | Checkpoint for `-b` mode |

---

## Architecture

### Address Mode
1. Batch point computation: P = A + i*G, Q = A − i*G
2. Montgomery batch inversion (one inverse per 2048 points)
3. SHA-256 + RIPEMD-160 → hash160
4. Bloom filter pre-screen
5. Exact hash160 comparison

**File:** [`cgtgpu.cu:593`](cgtgpu.cu#L593)

### Pubkey Mode (x-coordinate only)
1. Same batch walk and inversion
2. **Skip SHA-256, RIPEMD-160, and y computation**
3. Shared-memory 64 Kbit screen (one probe per key)
4. Global bloom filter (8 probes if screen passes)
5. Direct x-coordinate comparison

**File:** [`cgtgpu.cu:875`](cgtgpu.cu#L875)

**Why It's Fast:**
- No hashing: 100% of address-mode's SHA-256/RIPEMD-160 cost eliminated
- No y arithmetic: `x(P) = λ² − ax − sx` never references y, saving 2 field multiplies per point
- Memory-optimized: shared-memory pre-screen stops 99.998% of keys before global memory access

---

## Tuning

Override defaults at build time:

```bash
make TUNE="-DCGT_STRIDE_HALF=512 -DCGT_ROUNDS=16"
```

| Define | Default | Description |
|---|---|---|
| `CGT_STRIDE_HALF` | 1024 | Batch size (points per lane) |
| `CGT_ROUNDS` | 8 | Rounds per kernel launch |
| `CGT_BLOCK_THREADS` | 512 | Threads per CUDA block |
| `CGT_BLOCKS_PER_SM` | 1 | Blocks per SM |

---

## Technical Details

### Elliptic Curve
- **Curve:** secp256k1
- **Field:** p = 2^256 − 0x1000003D1
- **Order:** n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

### Batch Inversion
Montgomery's trick:
```
d₀ = (x_step[0] − x_anchor)
d₁ = (x_step[1] − x_anchor)
...
d_n = (x_step[n] − x_anchor)

prefix[0] = d₀
prefix[i] = prefix[i−1] × dᵢ

acc = 1 / prefix[n]              // ONE field inverse

dᵢ⁻¹ = acc × prefix[i−1]
acc ← acc × dᵢ
```

One 255-squaring modular inverse amortized over 2048 point pairs.

### Symmetric Walk
P = A + i*G and Q = A − i*G share denominator (x_step − x_anchor):
```
λ_P = (y_step − y_anchor) / d
λ_Q = (−y_step − y_anchor) / d
```

Same inverse, two slopes, two x-coordinates → 9 field multiplies per pair (address mode) or 7 (pubkey mode).

---

## Verification

The engine independently confirms every GPU hit on the host before reporting:

**Address Mode:**
1. Rebuild scalar: `k = anchor ± offset`
2. Compute `P = k × G`
3. Serialize compressed/uncompressed
4. Hash160(P)
5. Compare against target

**Pubkey Mode:**
1. Rebuild scalar: `k = anchor ± offset`
2. Compute `P = k × G`
3. Compare `x(P)` against target x-coordinate
4. If `y(P) ≠ y(target)`, try mirror: `k' = n − k`

No false positives reach `cgtkey_found.txt`.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Batch inversion:** Montgomery, "Speeding the Pollard and Elliptic Curve Methods of Factorization" (1987)
- **Symmetric walk optimization:** Observed independently; common in Pollard rho implementations
- **Kirsch-Mitzenmacher double hashing:** "Less Hashing, Same Performance: Building a Better Bloom Filter" (2006)

---

## Contact

**Author:** Cryptographytube  
**GitHub:** [https://github.com/Cryptographytube](https://github.com/Cryptographytube)

---

## Changelog

### v1.0 (2026-08-06)
- Initial release
- Dual-mode architecture: address (hash160) + pubkey (x-coordinate)
- 10,000 Mkey/s pubkey search on RTX 5070 Ti
- Perfect-coverage random mode with resume support
- Fat binary: sm_50 through sm_120+
