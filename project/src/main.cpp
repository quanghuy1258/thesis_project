#include "mpc_application.h"

void save_data(const std::string &fileName, void *buffer, int sz) {
  std::ofstream f(fileName, std::ifstream::binary);
  f.write((char *)buffer, sz);
  f.close();
}
void load_data(const std::string &fileName, void *buffer, int sz) {
  std::ifstream f(fileName, std::ifstream::binary);
  f.read((char *)buffer, sz);
  f.close();
}

// Simple circuit for debugging
void xorAll(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
            std::vector<thesis::TrlweCipher *> &out, MpcApplication &party) {
  for (size_t i = 0; i < out.size(); i++) {
    out[i] = party.reduce(inp[0][i]);
    for (size_t j = 1; j < inp.size(); j++) {
      auto cipher_inp = party.reduce(inp[j][i]);
      auto cipher_out = out[i];
      out[i] = party.addOp(cipher_out, cipher_inp);
      delete cipher_inp;
      delete cipher_out;
    }
  }
}

// Demo
int main(int argc, char *argv[]) {
#ifdef USING_32BIT
  std::cerr << "ERROR: Please run with 64-bit mode" << std::endl;
  return 1;
#endif
  if (argc != 3) {
    std::cerr << "Help: " << std::endl;
    std::cerr << "  " << argv[0] << " <id_party> <plaintext>" << std::endl;
    return 1;
  }
  // Set params
  int numParty = 3;
  int N = 1024;
  int m = 6;
  int l = 64;
  double sdFresh = 1e-15;
  // Set id_party and plaintext
  int idParty = -1;
  int plaintext = 0;
  try {
    idParty = std::stoi(argv[1]);
  } catch (const std::invalid_argument &ia) {
    std::cerr << "Invalid argument: " << ia.what() << std::endl;
    std::cerr << "ERROR: id_party must be an integer" << std::endl;
  }
  try {
    plaintext = std::stoi(argv[2]) & 0xff;
  } catch (const std::invalid_argument &ia) {
    std::cerr << "Invalid argument: " << ia.what() << std::endl;
    std::cerr << "ERROR: plaintext must be an integer" << std::endl;
  }
  if (idParty < 0 || idParty >= numParty) {
    std::cerr << "ERROR: id_party is out of range" << std::endl;
    return 1;
  }
  std::cout << "INFO: Your plaintext is " << plaintext << std::endl;
  // Init party
  MpcApplication party(numParty, idParty, N, m, l, sdFresh);
  // Create private and public key
  party.createPrivkey();
  party.createPubkey();
  // Export public key
  {
    void *mem = std::malloc(party.getSizePubkey());
    party.exportPubkey(mem);
    std::string fileNameFormat = "pubkey_";
    save_data(fileNameFormat + argv[1], mem, party.getSizePubkey());
    std::free(mem);
  }
  // Wait for all parties to broadcast their public keys
  std::cout << "INFO: Wait for all parties to broadcast their public keys"
            << std::endl;
  std::cout << "INFO: Press ENTER to continue..." << std::endl;
  std::cin.get();
  // Create pre expand
  for (int i = 0; i < numParty; i++) {
    if (i == idParty)
      continue;
    void *mem_pub = std::malloc(party.getSizePubkey());
    void *mem_pre = std::malloc(party.getSizePreExpand());
    std::string fileName;
    fileName = "pubkey_";
    fileName += std::to_string(i);
    load_data(fileName, mem_pub, party.getSizePubkey());
    party.preExpand(mem_pub, mem_pre);
    fileName = "pre_expand_";
    fileName += std::to_string(i) + "_";
    fileName += argv[1];
    save_data(fileName, mem_pre, party.getSizePreExpand());
    std::free(mem_pub);
    std::free(mem_pre);
  }
  // Wait for all parties to broadcast their pre_expand
  std::cout << "INFO: Wait for all parties to broadcast their pre_expand"
            << std::endl;
  std::cout << "INFO: Press ENTER to continue..." << std::endl;
  std::cin.get();
  // Import all pre_expand
  std::vector<void *> pre_expand_list(numParty);
  for (int i = 0; i < numParty; i++) {
    if (i == idParty) {
      pre_expand_list[i] = nullptr;
      continue;
    }
    pre_expand_list[i] = std::malloc(party.getSizePreExpand());
    std::string fileName;
    fileName = "pre_expand_";
    fileName += argv[1];
    fileName += "_";
    fileName += std::to_string(i);
    load_data(fileName, pre_expand_list[i], party.getSizePubkey());
  }
  // Create ciphertexts and self-expand
  for (int i = 0; i < 8; i++) {
    bool msg = (plaintext >> i) & 1;
    void *main_cipher = std::malloc(party.getSizeMainCipher());
    void *random = std::malloc(party.getSizeRandom());
    void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
    party.encrypt(msg, main_cipher, nullptr, random);
    auto cipher = party.expandWithPlainRandom(pre_expand_list, nullptr, idParty,
                                              main_cipher, random);
    party.exportExpandedCipher(cipher, cipher_mem);
    std::string fileName;
    fileName = "cipher_";
    fileName += std::to_string(i) + "_";
    fileName += argv[1];
    save_data(fileName, cipher_mem, party.getSizeExpandedCipher());
    delete cipher;
    std::free(main_cipher);
    std::free(random);
    std::free(cipher_mem);
  }
  // Free all pre_expand
  for (int i = 0; i < numParty; i++) {
    if (i == idParty || !pre_expand_list[i])
      continue;
    std::free(pre_expand_list[i]);
    pre_expand_list[i] = nullptr;
  }
  // Wait for all parties to create and expand ciphertexts
  std::cout << "INFO: Wait for all parties to create and expand ciphertexts"
            << std::endl;
  std::cout << "INFO: Press ENTER to continue..." << std::endl;
  std::cin.get();
  // Import all input cipher
  std::vector<std::vector<thesis::TrgswCipher *>> cipher_list(numParty);
  for (int i = 0; i < numParty; i++) {
    cipher_list[i].resize(8, nullptr);
    for (int j = 0; j < 8; j++) {
      void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
      std::string fileName;
      fileName = "cipher_";
      fileName += std::to_string(j) + "_";
      fileName += std::to_string(i);
      load_data(fileName, cipher_mem, party.getSizeExpandedCipher());
      cipher_list[i][j] = party.importExpandedCipher(cipher_mem);
      std::free(cipher_mem);
    }
  }
  // Evaluation
  std::vector<thesis::TrlweCipher *> out_cipher_list(8, nullptr);
  xorAll(cipher_list, out_cipher_list, party);
  // Free all input cipher
  for (int i = 0; i < numParty; i++) {
    for (int j = 0; j < 8; j++) {
      if (cipher_list[i][j]) {
        delete cipher_list[i][j];
        cipher_list[i][j] = nullptr;
      }
    }
  }
  // Part decrypt and export result
  for (int i = 0; i < 8; i++) {
    thesis::TorusInteger out = party.partDec(out_cipher_list[i]);
    std::string fileName;
    fileName = "output_";
    fileName += std::to_string(i) + "_";
    fileName += argv[1];
    save_data(fileName, &out, sizeof(out));
  }
  // Free output cipher
  for (int i = 0; i < 8; i++) {
    if (out_cipher_list[i]) {
      delete out_cipher_list[i];
      out_cipher_list[i] = nullptr;
    }
  }
  // Wait for all parties to part decrypt
  std::cout << "INFO: Wait for all parties to part decrypt" << std::endl;
  std::cout << "INFO: Press ENTER to continue..." << std::endl;
  std::cin.get();
  // Final decrypt and print result
  int res = 0;
  for (int i = 7; i >= 0; i--) {
    std::vector<thesis::TorusInteger> out_list(numParty);
    std::string fileName;
    fileName = "output_";
    fileName += std::to_string(i) + "_";
    for (int j = 0; j < numParty; j++)
      load_data(fileName + std::to_string(j), &out_list[j],
                sizeof(out_list[j]));
    bool msg = party.finDec(out_list.data(), nullptr);
    res <<= 1;
    if (msg)
      res += 1;
  }
  std::cout << "INFO: Result is " << res << std::endl;
  return 0;
}
