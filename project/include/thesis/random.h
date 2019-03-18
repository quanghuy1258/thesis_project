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
  static void addSeed(uint32_t seed);
  static Torus getUniformTorus();
  static Torus getNormalTorus(double mean, double stddev);
  static double getErrorProbability(double stddev, double boundary);
  static Integer getUniformInteger();
};

} // namespace thesis

#endif
