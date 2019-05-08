#ifndef TORUS_UTILITY_H
#define TORUS_UTILITY_H

#include "thesis/declarations.h"
#include "thesis/load_lib.h"

namespace thesis {

class TorusUtility {
private:
  static void cudaAddVector(TorusInteger *dest, TorusInteger *src, int len,
                            void *streamPtr = nullptr);
  static void cudaSubVector(TorusInteger *dest, TorusInteger *src, int len,
                            void *streamPtr = nullptr);

public:
  static void addVector(TorusInteger *dest, TorusInteger *src, int len,
                        void *streamPtr = nullptr);
  static void subVector(TorusInteger *dest, TorusInteger *src, int len,
                        void *streamPtr = nullptr);
};

} // namespace thesis

#endif
