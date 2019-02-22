#include "thesis/random.h"

namespace thesis {

bool Random::_isInitSeeds = false;
std::vector<uint32_t> Random::_seeds{
    (unsigned)std::chrono::system_clock::now().time_since_epoch().count()};
std::default_random_engine Random::_generator;

void Random::initSeeds() {
  _isInitSeeds = true;
  std::seed_seq seed(_seeds.begin(), _seeds.end());
  _generator.seed(seed);
}
void Random::addSeed(const uint32_t &seed) {
  _isInitSeeds = false;
  _seeds.push_back(seed);
}

Torus Random::getUniformTorus() {
  if (!_isInitSeeds) {
    initSeeds();
  }
  std::uniform_int_distribution<Torus> distribution(
      std::numeric_limits<Torus>::min(), std::numeric_limits<Torus>::max());
  return distribution(_generator);
}
Torus Random::getNormalTorus(const double &mean, const double &stddev) {
  if (!_isInitSeeds) {
    initSeeds();
  }
  std::normal_distribution<double> distribution(mean, stddev);
  double randomNumber = distribution(_generator);
  randomNumber = (randomNumber - std::round(randomNumber)) *
                 std::pow(2, sizeof(Torus) * 8);
  return (Torus)randomNumber;
}
Integer Random::getUniformInteger() {
  if (!_isInitSeeds) {
    initSeeds();
  }
  std::uniform_int_distribution<Integer> distribution(
      std::numeric_limits<Integer>::min(), std::numeric_limits<Integer>::max());
  return distribution(_generator);
}

} // namespace thesis
