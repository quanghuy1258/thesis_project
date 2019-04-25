#ifndef THESIS_RANDOM_H
#define THESIS_RANDOM_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class Random {
private:
  static void initSeeds();

public:
  static void addSeed(unsigned seed);

  static void setUniform(TorusInteger *ptr, size_t len);
  static void setNormalTorus(TorusInteger *ptr, size_t len, double stddev);

  static double getErrorProbability(double stddev, double boundary);
};

} // namespace thesis

#endif
