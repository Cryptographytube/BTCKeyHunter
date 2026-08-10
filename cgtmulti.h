/* cgtmulti.h - Multi-GPU manager for CGTKEY. */
#ifndef CGTMULTI_H
#define CGTMULTI_H

#include "cgtgpu.h"
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>

/* Manager для работы с несколькими GPU одновременно.
 * Каждый GPU получает свой поддиапазон для поиска. */
class CgtMultiGpu {
public:
  CgtMultiGpu();
  ~CgtMultiGpu();

  /* Инициализировать указанные GPU. Если gpuIds пуст - использовать все доступные. */
  bool init(const std::vector<int> &gpuIds, std::string &err);

  /* Загрузить фильтр (bloom) на все устройства. */
  bool uploadFilter(const uint8_t *bloom, size_t bloomBytes,
                    uint64_t bits, int hashes, std::string &err);

  /* Загрузить фильтр для pubkey режима на все устройства. */
  bool uploadPubkeyFilter(const uint8_t *bloom, size_t bloomBytes,
                          uint64_t bits, int hashes,
                          const uint64_t *xCoords, size_t targetCount,
                          std::string &err);

  /* Настроить режимы (compressed/uncompressed) для всех устройств. */
  void setModes(bool compressed, bool uncompressed);

  /* Посеять якоря на всех устройствах с учётом смещения диапазона. */
  bool seedAnchors(const std::vector<cgt_u256> &baseKeys, 
                   const std::vector<cgt_u256> &offsets,
                   std::string &err);

  /* Посеять якоря в случайном режиме со смещением. */
  bool seedAnchorsRandom(const cgt_u256 &base, const std::vector<cgt_u256> &offsets,
                         std::string &err);

  /* Запустить один проход на всех GPU и собрать результаты. */
  bool runPass(std::vector<cgt_hit> &hits, std::string &err);

  /* Продвинуть якоря на всех устройствах. */
  bool advanceAnchors(std::string &err);

  /* Получить количество GPU. */
  int gpuCount() const { return (int)gpus.size(); }

  /* Получить информацию о каждом GPU. */
  const std::vector<cgt_devinfo> &getInfo() const { return info; }

  /* Общее количество ключей за один проход по всем GPU. */
  uint64_t totalKeysPerPass() const;

  /* Количество потоков (lanes) на каждом GPU. */
  const std::vector<int> &lanesPerGpu() const { return laneCounts; }

  /* Остановить все потоки поиска. */
  void stop();

private:
  std::vector<CgtGpu*> gpus;
  std::vector<cgt_devinfo> info;
  std::vector<int> laneCounts;
  std::vector<std::thread> workers;
  std::atomic<bool> running;
  std::mutex hitsMutex;

  /* Вспомогательная функция для запуска потока на одном GPU. */
  void workerThread(int gpuIndex, 
                    const cgt_u256 *rangeStart,
                    const cgt_u256 *rangeEnd,
                    bool randomMode,
                    std::vector<cgt_hit> &localHits,
                    std::string &localErr);
};

#endif /* CGTMULTI_H */
