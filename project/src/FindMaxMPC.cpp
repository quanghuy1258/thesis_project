#include "mpc_application.h"

#define NUM_BIT 5

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

// Max circuit - Packing
thesis::TrlweCipher *
maxPartyPacking(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                MpcApplication &party, size_t partyId, int N) {
  // Prepare params
  int N_bit = 0;
  while ((1 << (N_bit + 1)) <= N)
    N_bit++;
  int all_bit = inp.size() * NUM_BIT;
  int cipher_bit = (all_bit > N_bit) ? (all_bit - N_bit) : 0;
  // Prepare list of pseudo ciphertexts
  std::vector<thesis::TrlweCipher *> pseudoCipher_list(1 << cipher_bit,
                                                       nullptr);
  bool *msgPol = new bool[N];
  std::memset(msgPol, 0, N * sizeof(bool));
  std::vector<int> num(inp.size());
  for (size_t i = 0; i < pseudoCipher_list.size(); i++) {
    for (int j = 0; j < N; j++) {
      int val = i * N + j;
      if (val >= (1 << all_bit)) {
        msgPol[j] = false;
        continue;
      }
      for (size_t k = 0; k < inp.size(); k++)
        num[k] = (val >> (k * NUM_BIT)) & ((1 << NUM_BIT) - 1);
      msgPol[j] = true;
      for (size_t k = 0; k < inp.size(); k++) {
        if (k == partyId)
          continue;
        if (num[k] > num[partyId]) {
          msgPol[j] = false;
          break;
        }
      }
    }
    pseudoCipher_list[i] = party.pseudoCipher(msgPol);
  }
  delete[] msgPol;
  // Calculate
  for (int i = 0; i < cipher_bit; i++) {
    size_t x = (i + N_bit) / NUM_BIT;
    size_t y = (i + N_bit) % NUM_BIT;
    if (x >= inp.size())
      break;
    for (size_t j = 0; j < pseudoCipher_list.size(); j += 2) {
      if (pseudoCipher_list[j + 1] == nullptr)
        break;
      auto res = party.cMux(inp[x][y], pseudoCipher_list[j + 1],
                            pseudoCipher_list[j], j == 0);
      delete pseudoCipher_list[j];
      delete pseudoCipher_list[j + 1];
      pseudoCipher_list[j] = nullptr;
      pseudoCipher_list[j + 1] = nullptr;
      pseudoCipher_list[j / 2] = res;
    }
  }
  for (int i = 0; i < N_bit; i++) {
    size_t x = i / NUM_BIT;
    size_t y = i % NUM_BIT;
    if (x >= inp.size())
      break;
    auto temp = pseudoCipher_list[0];
    pseudoCipher_list[0] = party.blindRotate(temp, inp[x][y], -(1 << i));
    delete temp;
  }
  return pseudoCipher_list[0];
}
void maxAllPacking(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                   std::vector<thesis::TrlweCipher *> &out,
                   MpcApplication &party, int N) {
  for (size_t i = 0; i < out.size(); i++)
    out[i] = maxPartyPacking(inp, party, i, N);
}

// Demo
int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Help: " << std::endl;
    std::cerr << "  " << argv[0] << " <id_party> <plaintext>" << std::endl;
    return 1;
  }
  // Set params
  int numParty = 3;
  int N = 1024;
  int m = 6;
  int l = 32;
  double sdFresh = 1e-9;
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
    plaintext = std::stoi(argv[2]) & ((1 << NUM_BIT) - 1);
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
  for (int i = 0; i < NUM_BIT; i++) {
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
    cipher_list[i].resize(NUM_BIT, nullptr);
    for (int j = 0; j < NUM_BIT; j++) {
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
  std::vector<thesis::TrlweCipher *> out_cipher_list(numParty, nullptr);
  maxAllPacking(cipher_list, out_cipher_list, party, N);
  // Free all input cipher
  for (int i = 0; i < numParty; i++) {
    for (int j = 0; j < NUM_BIT; j++) {
      if (cipher_list[i][j]) {
        delete cipher_list[i][j];
        cipher_list[i][j] = nullptr;
      }
    }
  }
  // Part decrypt and export result
  for (int i = 0; i < numParty; i++) {
    thesis::TorusInteger out = party.partDec(out_cipher_list[i]);
    std::string fileName;
    fileName = "output_";
    fileName += std::to_string(i) + "_";
    fileName += argv[1];
    save_data(fileName, &out, sizeof(out));
  }
  // Free output cipher
  for (int i = 0; i < numParty; i++) {
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
  for (int i = 0; i < numParty; i++) {
    std::vector<thesis::TorusInteger> out_list(numParty);
    std::string fileName;
    fileName = "output_";
    fileName += std::to_string(i) + "_";
    for (int j = 0; j < numParty; j++)
      load_data(fileName + std::to_string(j), &out_list[j],
                sizeof(out_list[j]));
    bool msg = party.finDec(out_list.data(), nullptr);
    std::cout << "INFO: Does party " << i << " have the max value? ";
    if (msg)
      std::cout << "Yes" << std::endl;
    else
      std::cout << "No" << std::endl;
  }
  return 0;
}
