#include "thesis/random.h"

namespace thesis {

static std::mutex random_mtx;
static bool isInitSeeds = false;
static std::vector<unsigned> seeds{
    std::chrono::system_clock::now().time_since_epoch().count()};
static std::default_random_engine generator;

void Random::initSeeds() {
  isInitSeeds = true;
  std::seed_seq seed(seeds.begin(), seeds.end());
  generator.seed(seed);
}

void Random::addSeed(uint32_t seed) {
  std::lock_guard<std::mutex> guard(random_mtx);
  isInitSeeds = false;
  seeds.push_back(seed);
}

void Random::setUniformTorus(Torus *ptr, size_t len) {
  std::uniform_int_distribution<Torus> distribution(
      std::numeric_limits<Torus>::min(), std::numeric_limits<Torus>::max());
  std::lock_guard<std::mutex> guard(random_mtx);
  if (!isInitSeeds)
    initSeeds();
  for (size_t i = 0; i < len; i++)
    ptr[i] = distribution(generator);
}
void Random::setUniformInteger(Integer *ptr, size_t len) {
  std::uniform_int_distribution<Integer> distribution(
      std::numeric_limits<Integer>::min(), std::numeric_limits<Integer>::max());
  std::lock_guard<std::mutex> guard(random_mtx);
  if (!isInitSeeds)
    initSeeds();
  for (size_t i = 0; i < len; i++)
    ptr[i] = distribution(generator);
}
void Random::setNormalTorus(Torus *ptr, size_t len, double stddev) {
  const int bitsize_Torus = sizeof(Torus) * 8;
  stddev = std::abs(stddev);
  std::normal_distribution<double> distribution(0., stddev);
  std::lock_guard<std::mutex> guard(random_mtx);
  if (!isInitSeeds)
    initSeeds();
  for (size_t i = 0; i < len; i++) {
    double r = distribution(generator);
    ptr[i] = (r - std::round(r)) * std::pow(2, bitsize_Torus);
  }
}

double getErrorProbability(double stddev, double boundary) {
  stddev = std::abs(stddev);
  boundary = std::abs(boundary);
  if (stddev == 0)
    return -1;
  return std::erfc(boundary / (std::sqrt(2) * stddev));
}

Torus Random::getUniformTorus() {
  if (!isInitSeeds)
    initSeeds();
  std::uniform_int_distribution<Torus> distribution(
      std::numeric_limits<Torus>::min(), std::numeric_limits<Torus>::max());
  return distribution(generator);
}
Torus Random::getNormalTorus(double mean, double stddev) {
  if (!isInitSeeds)
    initSeeds();
  std::normal_distribution<double> distribution(mean, stddev);
  double randomNumber = distribution(generator);
  randomNumber = (randomNumber - std::round(randomNumber)) *
                 std::pow(2, sizeof(Torus) * 8);
  return (Torus)randomNumber;
}
Integer Random::getUniformInteger() {
  if (!isInitSeeds)
    initSeeds();
  std::uniform_int_distribution<Integer> distribution(
      std::numeric_limits<Integer>::min(), std::numeric_limits<Integer>::max());
  return distribution(generator);
}

} // namespace thesis
