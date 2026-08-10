/* cgtmulti.cpp - Multi-GPU manager implementation for CGTKEY. */
#include "cgtmulti.h"
#include "cgtmath.h"
#include <cstdio>
#include <cstring>

CgtMultiGpu::CgtMultiGpu() : running(false) {}

CgtMultiGpu::~CgtMultiGpu() {
  stop();
  for (size_t i = 0; i < gpus.size(); i++) {
    delete gpus[i];
  }
  gpus.clear();
}

bool CgtMultiGpu::init(const std::vector<int> &gpuIds, std::string &err) {
  std::vector<cgt_devinfo> allDevs;
  if (!cgtGpuList(allDevs)) {
    err = "No CUDA devices found";
    return false;
  }

  std::vector<int> ids = gpuIds;
  if (ids.empty()) {
    /* Use all available GPUs */
    ids.resize(allDevs.size());
    for (size_t i = 0; i < ids.size(); i++) ids[i] = (int)i;
  }

  /* Validate IDs */
  for (int id : ids) {
    if (id < 0 || id >= (int)allDevs.size()) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Invalid GPU ID %d (max %d)", id, (int)allDevs.size() - 1);
      err = buf;
      return false;
    }
  }

  gpus.resize(ids.size());
  info.resize(ids.size());
  laneCounts.resize(ids.size());

  for (size_t i = 0; i < ids.size(); i++) {
    gpus[i] = new CgtGpu();
    if (!gpus[i]->open(ids[i], err)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Failed to open GPU %d", ids[i]);
      err = buf;
      return false;
    }
    info[i] = gpus[i]->info();
    laneCounts[i] = gpus[i]->laneCount();
    printf("[+] GPU %zu: %s (lanes=%d)\n", i, info[i].name.c_str(), laneCounts[i]);
  }

  return true;
}

bool CgtMultiGpu::uploadFilter(const uint8_t *bloom, size_t bloomBytes,
                               uint64_t bits, int hashes, std::string &err) {
  for (size_t i = 0; i < gpus.size(); i++) {
    if (!gpus[i]->uploadFilter(bloom, bloomBytes, bits, hashes, err)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Failed to upload filter to GPU %zu", i);
      err = buf;
      return false;
    }
  }
  return true;
}

bool CgtMultiGpu::uploadPubkeyFilter(const uint8_t *bloom, size_t bloomBytes,
                                     uint64_t bits, int hashes,
                                     const uint64_t *xCoords, size_t targetCount,
                                     std::string &err) {
  for (size_t i = 0; i < gpus.size(); i++) {
    if (!gpus[i]->uploadPubkeyFilter(bloom, bloomBytes, bits, hashes, 
                                      xCoords, targetCount, err)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Failed to upload pubkey filter to GPU %zu", i);
      err = buf;
      return false;
    }
  }
  return true;
}

void CgtMultiGpu::setModes(bool compressed, bool uncompressed) {
  for (size_t i = 0; i < gpus.size(); i++) {
    gpus[i]->setModes(compressed, uncompressed);
  }
}

uint64_t CgtMultiGpu::totalKeysPerPass() const {
  uint64_t total = 0;
  for (size_t i = 0; i < gpus.size(); i++) {
    total += gpus[i]->keysPerPass();
  }
  return total;
}

bool CgtMultiGpu::seedAnchors(const std::vector<cgt_u256> &baseKeys,
                              const std::vector<cgt_u256> &offsets,
                              std::string &err) {
  if (baseKeys.size() != gpus.size() || offsets.size() != gpus.size()) {
    err = "Base keys and offsets must match GPU count";
    return false;
  }

  for (size_t i = 0; i < gpus.size(); i++) {
    std::vector<cgt_u256> laneKeys;
    laneKeys.reserve(gpus[i]->laneCount());

    /* Каждый GPU получает baseKeys[i] + offset для каждого lane */
    for (int lane = 0; lane < gpus[i]->laneCount(); lane++) {
      cgt_u256 key;
      cgtAdd(key, baseKeys[i], offsets[lane % offsets.size()]);
      laneKeys.push_back(key);
    }

    if (!gpus[i]->seedAnchors(laneKeys, err)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Failed to seed anchors on GPU %zu", i);
      err = buf;
      return false;
    }
  }
  return true;
}

bool CgtMultiGpu::seedAnchorsRandom(const cgt_u256 &base, 
                                    const std::vector<cgt_u256> &offsets,
                                    std::string &err) {
  if (offsets.empty()) {
    err = "Offsets cannot be empty";
    return false;
  }

  for (size_t i = 0; i < gpus.size(); i++) {
    /* Вычисляем базовую точку для этого GPU: base + globalOffset[i] */
    cgt_u256 gpuBase;
    cgtAdd(gpuBase, base, offsets[i % offsets.size()]);

    if (!gpus[i]->seedAnchorsRandom(gpuBase, err)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Failed to seed random anchors on GPU %zu", i);
      err = buf;
      return false;
    }
  }
  return true;
}

bool CgtMultiGpu::runPass(std::vector<cgt_hit> &hits, std::string &err) {
  std::vector<std::vector<cgt_hit>> localHits(gpus.size());
  std::vector<std::string> localErrs(gpus.size());
  std::vector<bool> success(gpus.size(), true);

  /* Запускаем все GPU параллельно */
  std::vector<std::thread> threads;
  for (size_t i = 0; i < gpus.size(); i++) {
    threads.emplace_back([this, i, &localHits, &localErrs, &success]() {
      localHits[i].clear();
      if (!gpus[i]->runPass(localHits[i], localErrs[i])) {
        success[i] = false;
      }
    });
  }

  /* Ждём завершения всех GPU */
  for (auto &t : threads) t.join();

  /* Собираем результаты */
  for (size_t i = 0; i < gpus.size(); i++) {
    if (!success[i]) {
      char buf[256];
      snprintf(buf, sizeof(buf), "GPU %zu failed: %s", i, localErrs[i].c_str());
      err = buf;
      return false;
    }
    hits.insert(hits.end(), localHits[i].begin(), localHits[i].end());
  }

  return true;
}

bool CgtMultiGpu::advanceAnchors(std::string &err) {
  for (size_t i = 0; i < gpus.size(); i++) {
    if (!gpus[i]->advanceAnchors(err)) {
      char buf[256];
      snprintf(buf, sizeof(buf), "Failed to advance anchors on GPU %zu", i);
      err = buf;
      return false;
    }
  }
  return true;
}

void CgtMultiGpu::stop() {
  running = false;
  /* workers будут проверять running и остановятся */
}
