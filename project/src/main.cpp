#include "thesis/load_lib.h"
#include "thesis/tfhe.h"

int main(int argc, char *argv[]) {
  std::cout << "Hello World!" << std::endl;
  for (int i = 0; i < argc; i++)
    std::cout << argv[i] << std::endl;
  thesis::Tfhe temp;
  return 0;
}
