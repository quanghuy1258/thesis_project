#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/tfhe.h"
#include "thesis/tlwe.h"
#include "thesis/torus.h"
#include "thesis/trgsw.h"
#include "thesis/trlwe.h"

int main(int argc, char *argv[]) {
  std::cout << "Hello World!" << std::endl;
  for (int i = 0; i < argc; i++)
    std::cout << argv[i] << std::endl;
  thesis::Tfhe temp_tfhe;
  thesis::Tlwe temp_tlwe;
  thesis::Torus temp_torus;
  return 0;
}
