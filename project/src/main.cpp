#include "thesis/load_lib.h"
#include "thesis/declarations.h"
#include "thesis/tfhe.h"
#include "thesis/tlwe.h"
#include "thesis/trlwe.h"
#include "thesis/tgsw.h"
#include "thesis/trgsw.h"

int main(int argc, char *argv[]) {
  std::cout << "Hello World!" << std::endl;
  for (int i = 0; i < argc; i++)
    std::cout << argv[i] << std::endl;
  thesis::Tfhe temp;
  return 0;
}
