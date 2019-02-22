#ifndef THESIS_RANDOM_H
#define THESIS_RANDOM_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Random {
private:
  static bool _isInitSeeds;
  static std::vector<uint32_t> _seeds;
  static std::default_random_engine _generator;

  static void initSeeds();

public:
  static void addSeed(const uint32_t &seed);
  static Torus getUniformTorus();
  static Torus getNormalTorus(const double &mean, const double &stddev);
  static Integer getUniformInteger();
};

} // namespace thesis

#endif
