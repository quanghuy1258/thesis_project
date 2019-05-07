#include "thesis/memory_management.h"
#include "thesis/random.h"

namespace thesis {

static std::mutex random_mtx;
static bool isInitSeeds = false;
static std::vector<unsigned> seeds{
    (unsigned)std::chrono::system_clock::now().time_since_epoch().count()};
static std::default_random_engine generator;

void Random::initSeeds() {
  isInitSeeds = true;
  std::seed_seq seed(seeds.begin(), seeds.end());
  generator.seed(seed);
}

void Random::addSeed(unsigned seed) {
  std::lock_guard<std::mutex> guard(random_mtx);
  isInitSeeds = false;
  seeds.push_back(seed);
}

void Random::setUniform(TorusInteger *ptr, size_t len,
                        std::function<TorusInteger(TorusInteger)> transformFn) {
  std::uniform_int_distribution<TorusInteger> distribution(
      std::numeric_limits<TorusInteger>::min(),
      std::numeric_limits<TorusInteger>::max());
  std::vector<TorusInteger> vec(len);
  {
    std::lock_guard<std::mutex> guard(random_mtx);
    if (!isInitSeeds)
      initSeeds();
    for (size_t i = 0; i < len; i++)
      vec[i] = transformFn(distribution(generator));
  }
  MemoryManagement::memcpyMM_h2d(ptr, vec.data(), len * sizeof(TorusInteger));
}
void Random::setNormalTorus(TorusInteger *ptr, size_t len, double stddev) {
  const int bitsize_TorusInteger = sizeof(TorusInteger) * 8;
  stddev = std::abs(stddev);
  std::normal_distribution<double> distribution(0., stddev);
  std::vector<TorusInteger> vec(len);
  {
    std::lock_guard<std::mutex> guard(random_mtx);
    if (!isInitSeeds)
      initSeeds();
    for (size_t i = 0; i < len; i++) {
      double r = distribution(generator);
      vec[i] = (r - std::round(r)) * std::pow(2, bitsize_TorusInteger);
    }
  }
  MemoryManagement::memcpyMM_h2d(ptr, vec.data(), len * sizeof(TorusInteger));
}

double getErrorProbability(double stddev, double boundary) {
  stddev = std::abs(stddev);
  boundary = std::abs(boundary);
  if (stddev <= 0 || boundary <= 0)
    return -1;
  return std::erfc(boundary / (std::sqrt(2) * stddev));
}

} // namespace thesis
