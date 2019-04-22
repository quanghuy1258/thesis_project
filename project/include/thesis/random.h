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

  static void setUniformTorus(Torus *ptr, size_t len);
  static void setUniformInteger(Integer *ptr, size_t len);
  static void setNormalTorus(Torus *ptr, size_t len, double stddev);

  static double getErrorProbability(double stddev, double boundary);

  [[deprecated("Will be removed")]]
  static Torus getUniformTorus();
  [[deprecated("Will be removed")]]
  static Torus getNormalTorus(double mean, double stddev);
  [[deprecated("Will be removed")]]
  static Integer getUniformInteger();
};

} // namespace thesis

#endif
