#include "mpc_application.h"
#include "thesis/profiling_timer.h"

#define NUM_BIT 5
#define PARAM_L 64
#define ALPHA 1e-16

//#define TEMPORARY
#define PACKING_SELF_EXPAND
//#define PACKING_EXPAND
//#define NON_PACKING_SELF_EXPAND
//#define NON_PACKING_EXPAND

void save_data(const std::string &fileName, void *buffer, int sz) {
  std::ofstream f(fileName, std::ifstream::binary);
  if (!f.good()) {
    std::cerr << "ERROR: File Problem --> " << fileName << std::endl;
    throw std::runtime_error("ERROR: File Problem");
  }
  f.write((char *)buffer, sz);
  f.close();
}
void load_data(const std::string &fileName, void *buffer, int sz) {
  std::ifstream f(fileName, std::ifstream::binary);
  if (!f.good()) {
    std::cerr << "ERROR: File Problem --> " << fileName << std::endl;
    throw std::runtime_error("ERROR: File Problem");
  }
  f.read((char *)buffer, sz);
  f.close();
}
bool check_data(const std::string &fileName) {
  std::ifstream f(fileName);
  return f.good();
}

#if defined(TEMPORARY)
// Simple circuit for testing
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

// Main
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
  int l = PARAM_L;
  double sdFresh = ALPHA;
  // Set id_party and plaintext
  int idParty = -1;
  int plaintext = -1;
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
  if (idParty < 0 || idParty >= numParty || plaintext < 0) {
    std::cerr << "ERROR: id_party is out of range" << std::endl;
    return 1;
  }
  std::cout << "INFO: Your plaintext is " << plaintext << std::endl;
  // Before Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    std::string fileNameFormat = "pubkey_";
    if (check_data(fileNameFormat + std::to_string(idParty - 1)))
      break;
  }
  {
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
    // Export private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      party.exportPrivkey(mem);
      std::string fileNameFormat = "privkey_";
      save_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      std::free(mem);
    }
    // Wait for all parties to broadcast their public keys
    std::cout << "INFO: Wait for all parties to broadcast their public keys"
              << std::endl;
    while (true) {
      bool brk = true;
      std::string fileNameFormat = "pubkey_";
      for (int i = 0; i < numParty; i++) {
        if (!check_data(fileNameFormat + std::to_string(i))) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < numParty; i++) {
        if (i == idParty - 1)
          continue;
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
    while (true) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; (i < numParty) && brk; i++) {
        for (int j = 0; j < numParty; j++) {
          if (i == j)
            continue;
          fileName = "pre_expand_";
          fileName += std::to_string(i) + "_";
          fileName += std::to_string(j);
          if (!check_data(fileName)) {
            brk = false;
            break;
          }
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < NUM_BIT; i++) {
        fileName = "cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
      load_data(fileName, pre_expand_list[i], party.getSizePreExpand());
    }
    // Create ciphertexts and self-expand
    for (int i = 0; i < NUM_BIT; i++) {
      bool msg = (plaintext >> i) & 1;
      void *main_cipher = std::malloc(party.getSizeMainCipher());
      void *random = std::malloc(party.getSizeRandom());
      void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
      party.encrypt(msg, main_cipher, nullptr, random);
      auto cipher = party.expandWithPlainRandom(pre_expand_list, nullptr,
                                                idParty, main_cipher, random);
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
  }
  // Wait for all parties to create and expand ciphertexts
  std::cout << "INFO: Wait for all parties to create and expand ciphertexts"
            << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < NUM_BIT) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  // During Eval phase
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
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
    std::vector<thesis::TrlweCipher *> out_cipher_list(NUM_BIT, nullptr);
    xorAll(cipher_list, out_cipher_list, party);
    // Free all input cipher
    for (int i = 0; i < numParty; i++) {
      for (int j = 0; j < NUM_BIT; j++) {
        if (cipher_list[i][j]) {
          delete cipher_list[i][j];
          cipher_list[i][j] = nullptr;
        }
      }
    }
    // Export output cipher
    for (int i = 0; i < NUM_BIT; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      party.exportReducedCipher(out_cipher_list[i], output_cipher_mem);
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      save_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      std::free(output_cipher_mem);
    }
    // Free output cipher
    for (int i = 0; i < NUM_BIT; i++) {
      if (out_cipher_list[i]) {
        delete out_cipher_list[i];
        out_cipher_list[i] = nullptr;
      }
    }
  }
  // After Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; i < NUM_BIT; i++) {
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      fileName += std::to_string(idParty - 1);
      if (!check_data(fileName)) {
        brk = false;
        break;
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import public key
    {
      void *mem = std::malloc(party.getSizePubkey());
      std::string fileNameFormat = "pubkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePubkey());
      party.importPubkey(mem);
      std::free(mem);
    }
    // Import private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      std::string fileNameFormat = "privkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      party.importPrivkey(mem);
      std::free(mem);
    }
    // Import output cipher
    std::vector<thesis::TrlweCipher *> out_cipher_list(NUM_BIT, nullptr);
    for (int i = 0; i < NUM_BIT; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      load_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      out_cipher_list[i] = party.importReducedCipher(output_cipher_mem);
      std::free(output_cipher_mem);
    }
    // Part decrypt and export result
    for (int i = 0; i < NUM_BIT; i++) {
      thesis::TorusInteger out = party.partDec(out_cipher_list[i]);
      std::string fileName;
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      fileName += argv[1];
      save_data(fileName, &out, sizeof(out));
    }
    // Free output cipher
    for (int i = 0; i < NUM_BIT; i++) {
      if (out_cipher_list[i]) {
        delete out_cipher_list[i];
        out_cipher_list[i] = nullptr;
      }
    }
  }
  // Wait for all parties to part decrypt
  std::cout << "INFO: Wait for all parties to part decrypt" << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < NUM_BIT) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "output_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Final decrypt and print result
    int res = 0;
    for (int i = NUM_BIT - 1; i >= 0; i--) {
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
  }
  return 0;
}
#elif defined(PACKING_SELF_EXPAND)
// Max circuit - Packing
thesis::TrlweCipher *
maxPartyPacking(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                MpcApplication &party, size_t partyId, int N) {
  DECLARE_TIMING(Debug);
  // Prepare params
  START_TIMING(Debug);
  int N_bit = 0;
  while ((1 << (N_bit + 1)) <= N)
    N_bit++;
  int all_bit = inp.size() * NUM_BIT;
  int cipher_bit = (all_bit > N_bit) ? (all_bit - N_bit) : 0;
  STOP_TIMING(Debug);
  PRINT_TIMING(Debug);
  // Prepare list of pseudo ciphertexts
  START_TIMING(Debug);
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
  STOP_TIMING(Debug);
  PRINT_TIMING(Debug);
  // Calculate
  START_TIMING(Debug);
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
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << pseudoCipher_list[0]->_sdError << " "
            << std::sqrt(pseudoCipher_list[0]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  START_TIMING(Debug);
  for (int i = 0; i < N_bit; i++) {
    size_t x = i / NUM_BIT;
    size_t y = i % NUM_BIT;
    if (x >= inp.size())
      break;
    auto temp = pseudoCipher_list[0];
    pseudoCipher_list[0] = party.blindRotate(temp, inp[x][y], -(1 << i));
    delete temp;
  }
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << pseudoCipher_list[0]->_sdError << " "
            << std::sqrt(pseudoCipher_list[0]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  return pseudoCipher_list[0];
}
void maxAllPacking(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                   std::vector<thesis::TrlweCipher *> &out,
                   MpcApplication &party, int N) {
  DECLARE_TIMING(Full);
  START_TIMING(Full);
  for (size_t i = 0; i < out.size(); i++)
    out[i] = maxPartyPacking(inp, party, i, N);
  STOP_TIMING(Full);
  PRINT_TIMING(Full);
}

// Main
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
  int l = PARAM_L;
  double sdFresh = ALPHA;
  // Set id_party and plaintext
  int idParty = -1;
  int plaintext = -1;
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
  if (idParty < 0 || idParty >= numParty || plaintext < 0) {
    std::cerr << "ERROR: id_party is out of range" << std::endl;
    return 1;
  }
  std::cout << "INFO: Your plaintext is " << plaintext << std::endl;
  // Before Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    std::string fileNameFormat = "pubkey_";
    if (check_data(fileNameFormat + std::to_string(idParty - 1)))
      break;
  }
  {
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
    // Export private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      party.exportPrivkey(mem);
      std::string fileNameFormat = "privkey_";
      save_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      std::free(mem);
    }
    // Wait for all parties to broadcast their public keys
    std::cout << "INFO: Wait for all parties to broadcast their public keys"
              << std::endl;
    while (true) {
      bool brk = true;
      std::string fileNameFormat = "pubkey_";
      for (int i = 0; i < numParty; i++) {
        if (!check_data(fileNameFormat + std::to_string(i))) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < numParty; i++) {
        if (i == idParty - 1)
          continue;
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
    while (true) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; (i < numParty) && brk; i++) {
        for (int j = 0; j < numParty; j++) {
          if (i == j)
            continue;
          fileName = "pre_expand_";
          fileName += std::to_string(i) + "_";
          fileName += std::to_string(j);
          if (!check_data(fileName)) {
            brk = false;
            break;
          }
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < NUM_BIT; i++) {
        fileName = "cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
      load_data(fileName, pre_expand_list[i], party.getSizePreExpand());
    }
    // Create ciphertexts and self-expand
    for (int i = 0; i < NUM_BIT; i++) {
      bool msg = (plaintext >> i) & 1;
      void *main_cipher = std::malloc(party.getSizeMainCipher());
      void *random = std::malloc(party.getSizeRandom());
      void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
      party.encrypt(msg, main_cipher, nullptr, random);
      auto cipher = party.expandWithPlainRandom(pre_expand_list, nullptr,
                                                idParty, main_cipher, random);
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
  }
  // Wait for all parties to create and expand ciphertexts
  std::cout << "INFO: Wait for all parties to create and expand ciphertexts"
            << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < NUM_BIT) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  // During Eval phase
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
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
    // Export output cipher
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      party.exportReducedCipher(out_cipher_list[i], output_cipher_mem);
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      save_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      std::free(output_cipher_mem);
    }
    // Free output cipher
    for (int i = 0; i < numParty; i++) {
      if (out_cipher_list[i]) {
        delete out_cipher_list[i];
        out_cipher_list[i] = nullptr;
      }
    }
  }
  // After Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; i < numParty; i++) {
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      fileName += std::to_string(idParty - 1);
      if (!check_data(fileName)) {
        brk = false;
        break;
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import public key
    {
      void *mem = std::malloc(party.getSizePubkey());
      std::string fileNameFormat = "pubkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePubkey());
      party.importPubkey(mem);
      std::free(mem);
    }
    // Import private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      std::string fileNameFormat = "privkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      party.importPrivkey(mem);
      std::free(mem);
    }
    // Import output cipher
    std::vector<thesis::TrlweCipher *> out_cipher_list(numParty, nullptr);
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      load_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      out_cipher_list[i] = party.importReducedCipher(output_cipher_mem);
      std::free(output_cipher_mem);
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
  }
  // Wait for all parties to part decrypt
  std::cout << "INFO: Wait for all parties to part decrypt" << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < numParty) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "output_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Final decrypt and print result
    for (int i = 0; i < numParty; i++) {
      std::vector<thesis::TorusInteger> out_list(numParty);
      std::string fileName;
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      for (int j = 0; j < numParty; j++)
        load_data(fileName + std::to_string(j), &out_list[j],
                  sizeof(out_list[j]));
      double debugError = 0;
      bool msg = party.finDec(out_list.data(), &debugError);
      std::cout << "INFO: Does party " << i << " have the max value? ";
      if (msg)
        std::cout << "Yes" << std::endl;
      else
        std::cout << "No" << std::endl;
      std::cout << "Debug: " << debugError << std::endl;
    }
  }
  return 0;
}
#elif defined(PACKING_EXPAND)
// Max circuit - Packing
thesis::TrlweCipher *
maxPartyPacking(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                MpcApplication &party, size_t partyId, int N) {
  DECLARE_TIMING(Debug);
  // Prepare params
  START_TIMING(Debug);
  int N_bit = 0;
  while ((1 << (N_bit + 1)) <= N)
    N_bit++;
  int all_bit = inp.size() * NUM_BIT;
  int cipher_bit = (all_bit > N_bit) ? (all_bit - N_bit) : 0;
  STOP_TIMING(Debug);
  PRINT_TIMING(Debug);
  // Prepare list of pseudo ciphertexts
  START_TIMING(Debug);
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
  STOP_TIMING(Debug);
  PRINT_TIMING(Debug);
  // Calculate
  START_TIMING(Debug);
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
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << pseudoCipher_list[0]->_sdError << " "
            << std::sqrt(pseudoCipher_list[0]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  START_TIMING(Debug);
  for (int i = 0; i < N_bit; i++) {
    size_t x = i / NUM_BIT;
    size_t y = i % NUM_BIT;
    if (x >= inp.size())
      break;
    auto temp = pseudoCipher_list[0];
    pseudoCipher_list[0] = party.blindRotate(temp, inp[x][y], -(1 << i));
    delete temp;
  }
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << pseudoCipher_list[0]->_sdError << " "
            << std::sqrt(pseudoCipher_list[0]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  return pseudoCipher_list[0];
}
void maxAllPacking(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                   std::vector<thesis::TrlweCipher *> &out,
                   MpcApplication &party, int N) {
  DECLARE_TIMING(Full);
  START_TIMING(Full);
  for (size_t i = 0; i < out.size(); i++)
    out[i] = maxPartyPacking(inp, party, i, N);
  STOP_TIMING(Full);
  PRINT_TIMING(Full);
}

// Main
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
  int l = PARAM_L;
  double sdFresh = ALPHA;
  // Set id_party and plaintext
  int idParty = -1;
  int plaintext = -1;
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
  if (idParty < 0 || idParty >= numParty || plaintext < 0) {
    std::cerr << "ERROR: id_party is out of range" << std::endl;
    return 1;
  }
  std::cout << "INFO: Your plaintext is " << plaintext << std::endl;
  // Before Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    std::string fileNameFormat = "pubkey_";
    if (check_data(fileNameFormat + std::to_string(idParty - 1)))
      break;
  }
  {
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
    // Export private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      party.exportPrivkey(mem);
      std::string fileNameFormat = "privkey_";
      save_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      std::free(mem);
    }
    // Wait for all parties to broadcast their public keys
    std::cout << "INFO: Wait for all parties to broadcast their public keys"
              << std::endl;
    while (true) {
      bool brk = true;
      std::string fileNameFormat = "pubkey_";
      for (int i = 0; i < numParty; i++) {
        if (!check_data(fileNameFormat + std::to_string(i))) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < numParty; i++) {
        if (i == idParty - 1)
          continue;
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
    while (true) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; (i < numParty) && brk; i++) {
        for (int j = 0; j < numParty; j++) {
          if (i == j)
            continue;
          fileName = "pre_expand_";
          fileName += std::to_string(i) + "_";
          fileName += std::to_string(j);
          if (!check_data(fileName)) {
            brk = false;
            break;
          }
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < NUM_BIT; i++) {
        fileName = "main_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
        fileName = "rand_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Create ciphertexts for msg and rand
    for (int i = 0; i < NUM_BIT; i++) {
      bool msg = (plaintext >> i) & 1;
      void *main_cipher = std::malloc(party.getSizeMainCipher());
      void *rand_cipher = std::malloc(party.getSizeRandCipher());
      party.encrypt(msg, main_cipher, rand_cipher, nullptr);
      std::string fileName;
      fileName = "main_cipher_";
      fileName += std::to_string(i) + "_";
      fileName += argv[1];
      save_data(fileName, main_cipher, party.getSizeMainCipher());
      fileName = "rand_cipher_";
      fileName += std::to_string(i) + "_";
      fileName += argv[1];
      save_data(fileName, rand_cipher, party.getSizeRandCipher());
      std::free(main_cipher);
      std::free(rand_cipher);
    }
  }
  // Wait for all parties to create and expand ciphertexts
  std::cout
      << "INFO: Wait for all parties to create ciphertexts for msg and rand"
      << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < NUM_BIT) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "main_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
        fileName = "rand_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  // Expand ciphertexts
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import all main_cipher, rand_cipher --> Expand --> Export expanded
    // ciphertexts
    void *main_cipher = std::malloc(party.getSizeMainCipher());
    void *rand_cipher = std::malloc(party.getSizeRandCipher());
    void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
    for (int i = 0; i < numParty; i++) {
      std::string fileName;
      std::vector<void *> pre_expand_list(numParty);
      for (int j = 0; j < numParty; j++) {
        if (j == i) {
          pre_expand_list[j] = nullptr;
          continue;
        }
        pre_expand_list[j] = std::malloc(party.getSizePreExpand());
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        load_data(fileName, pre_expand_list[j], party.getSizePreExpand());
      }
      for (int j = 0; j < NUM_BIT; j++) {
        fileName = "main_cipher_";
        fileName += std::to_string(j) + "_";
        fileName += std::to_string(i);
        load_data(fileName, main_cipher, party.getSizeMainCipher());
        fileName = "rand_cipher_";
        fileName += std::to_string(j) + "_";
        fileName += std::to_string(i);
        load_data(fileName, rand_cipher, party.getSizeRandCipher());
        thesis::TrgswCipher *cipher =
            party.expand(pre_expand_list, nullptr, i, main_cipher, rand_cipher);
        party.exportExpandedCipher(cipher, cipher_mem);
        fileName = "cipher_";
        fileName += std::to_string(j) + "_";
        fileName += std::to_string(i);
        save_data(fileName, cipher_mem, party.getSizeExpandedCipher());
      }
      for (int j = 0; j < numParty; j++) {
        if (pre_expand_list[j]) {
          std::free(pre_expand_list[j]);
          pre_expand_list[j] = nullptr;
        }
      }
    }
    std::free(main_cipher);
    std::free(rand_cipher);
    std::free(cipher_mem);
  }
  // During Eval phase
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
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
    // Export output cipher
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      party.exportReducedCipher(out_cipher_list[i], output_cipher_mem);
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      save_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      std::free(output_cipher_mem);
    }
    // Free output cipher
    for (int i = 0; i < numParty; i++) {
      if (out_cipher_list[i]) {
        delete out_cipher_list[i];
        out_cipher_list[i] = nullptr;
      }
    }
  }
  // After Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; i < numParty; i++) {
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      fileName += std::to_string(idParty - 1);
      if (!check_data(fileName)) {
        brk = false;
        break;
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import public key
    {
      void *mem = std::malloc(party.getSizePubkey());
      std::string fileNameFormat = "pubkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePubkey());
      party.importPubkey(mem);
      std::free(mem);
    }
    // Import private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      std::string fileNameFormat = "privkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      party.importPrivkey(mem);
      std::free(mem);
    }
    // Import output cipher
    std::vector<thesis::TrlweCipher *> out_cipher_list(numParty, nullptr);
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      load_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      out_cipher_list[i] = party.importReducedCipher(output_cipher_mem);
      std::free(output_cipher_mem);
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
  }
  // Wait for all parties to part decrypt
  std::cout << "INFO: Wait for all parties to part decrypt" << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < numParty) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "output_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Final decrypt and print result
    for (int i = 0; i < numParty; i++) {
      std::vector<thesis::TorusInteger> out_list(numParty);
      std::string fileName;
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      for (int j = 0; j < numParty; j++)
        load_data(fileName + std::to_string(j), &out_list[j],
                  sizeof(out_list[j]));
      double debugError = 0;
      bool msg = party.finDec(out_list.data(), &debugError);
      std::cout << "INFO: Does party " << i << " have the max value? ";
      if (msg)
        std::cout << "Yes" << std::endl;
      else
        std::cout << "No" << std::endl;
      std::cout << "Debug: " << debugError << std::endl;
    }
  }
  return 0;
}
#elif defined(NON_PACKING_SELF_EXPAND)
// Max circuit
void maxPairParty(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                  std::vector<thesis::TrlweCipher *> &out,
                  MpcApplication &party, int partyA, int partyB) {
  DECLARE_TIMING(Debug);
  START_TIMING(Debug);
  {
    auto notA = party.notOp(inp[partyA][NUM_BIT - 1]);
    auto BAndNotA = party.mulOp(inp[partyB][NUM_BIT - 1], notA);
    auto notBOrA = party.notOp(BAndNotA);
    if (out[partyA]) {
      auto temp = out[partyA];
      out[partyA] = party.mulOp(temp, notBOrA);
      delete temp;
    } else
      out[partyA] = party.reduce(notBOrA);
    delete notA;
    delete BAndNotA;
    delete notBOrA;
  }
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << partyA << " " << partyB << std::endl;
  std::cerr << "Debug: " << out[partyA]->_sdError << " "
            << std::sqrt(out[partyA]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  START_TIMING(Debug);
  {
    auto notB = party.notOp(inp[partyB][NUM_BIT - 1]);
    auto AAndNotB = party.mulOp(inp[partyA][NUM_BIT - 1], notB);
    auto notAOrB = party.notOp(AAndNotB);
    if (out[partyB]) {
      auto temp = out[partyB];
      out[partyB] = party.mulOp(temp, notAOrB);
      delete temp;
    } else
      out[partyB] = party.reduce(notAOrB);
    delete notB;
    delete AAndNotB;
    delete notAOrB;
  }
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << partyA << " " << partyB << std::endl;
  std::cerr << "Debug: " << out[partyB]->_sdError << " "
            << std::sqrt(out[partyB]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  thesis::TrgswCipher *Z = nullptr;
  for (int i = NUM_BIT - 1; i >= 1; i--) {
    START_TIMING(Debug);
    auto notAxorB = party.notXorOp(inp[partyA][i], inp[partyB][i]);
    if (Z) {
      auto temp = Z;
      Z = party.mulOp(temp, notAxorB);
      delete temp;
      temp = Z;
      Z = party.mulOp(temp, notAxorB);
      delete temp;
    } else
      Z = party.mulOp(notAxorB, notAxorB);
    delete notAxorB;
    STOP_TIMING(Debug);
    std::cerr << "Debug: " << partyA << " " << partyB << " " << i << std::endl;
    PRINT_TIMING(Debug);
    START_TIMING(Debug);
    {
      auto X = party.mulOp(Z, inp[partyB][i - 1]);
      auto temp = X;
      auto notA = party.notOp(inp[partyA][i - 1]);
      X = party.mulOp(temp, notA);
      delete temp;
      delete notA;
      temp = X;
      X = party.notOp(temp);
      delete temp;
      auto trlwe_temp = out[partyA];
      out[partyA] = party.mulOp(trlwe_temp, X);
      delete X;
      delete trlwe_temp;
    }
    STOP_TIMING(Debug);
    std::cerr << "Debug: " << partyA << " " << partyB << " " << i << std::endl;
    std::cerr << "Debug: " << out[partyA]->_sdError << " "
              << std::sqrt(out[partyA]->_varError) << std::endl;
    PRINT_TIMING(Debug);
    START_TIMING(Debug);
    {
      auto X = party.mulOp(Z, inp[partyA][i - 1]);
      auto temp = X;
      auto notB = party.notOp(inp[partyB][i - 1]);
      X = party.mulOp(temp, notB);
      delete temp;
      delete notB;
      temp = X;
      X = party.notOp(temp);
      delete temp;
      auto trlwe_temp = out[partyB];
      out[partyB] = party.mulOp(trlwe_temp, X);
      delete X;
      delete trlwe_temp;
    }
    STOP_TIMING(Debug);
    std::cerr << "Debug: " << partyA << " " << partyB << " " << i << std::endl;
    std::cerr << "Debug: " << out[partyB]->_sdError << " "
              << std::sqrt(out[partyB]->_varError) << std::endl;
    PRINT_TIMING(Debug);
  }
  delete Z;
  std::cerr << "Debug: " << partyA << " " << partyB << std::endl;
}
void maxAll(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
            std::vector<thesis::TrlweCipher *> &out, MpcApplication &party) {
  DECLARE_TIMING(Full);
  START_TIMING(Full);
  for (size_t i = 0; i < out.size(); i++) {
    for (size_t j = i + 1; j < out.size(); j++)
      maxPairParty(inp, out, party, i, j);
  }
  STOP_TIMING(Full);
  PRINT_TIMING(Full);
}

// Main
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
  int l = PARAM_L;
  double sdFresh = ALPHA;
  // Set id_party and plaintext
  int idParty = -1;
  int plaintext = -1;
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
  if (idParty < 0 || idParty >= numParty || plaintext < 0) {
    std::cerr << "ERROR: id_party is out of range" << std::endl;
    return 1;
  }
  std::cout << "INFO: Your plaintext is " << plaintext << std::endl;
  // Before Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    std::string fileNameFormat = "pubkey_";
    if (check_data(fileNameFormat + std::to_string(idParty - 1)))
      break;
  }
  {
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
    // Export private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      party.exportPrivkey(mem);
      std::string fileNameFormat = "privkey_";
      save_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      std::free(mem);
    }
    // Wait for all parties to broadcast their public keys
    std::cout << "INFO: Wait for all parties to broadcast their public keys"
              << std::endl;
    while (true) {
      bool brk = true;
      std::string fileNameFormat = "pubkey_";
      for (int i = 0; i < numParty; i++) {
        if (!check_data(fileNameFormat + std::to_string(i))) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < numParty; i++) {
        if (i == idParty - 1)
          continue;
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
    while (true) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; (i < numParty) && brk; i++) {
        for (int j = 0; j < numParty; j++) {
          if (i == j)
            continue;
          fileName = "pre_expand_";
          fileName += std::to_string(i) + "_";
          fileName += std::to_string(j);
          if (!check_data(fileName)) {
            brk = false;
            break;
          }
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < NUM_BIT; i++) {
        fileName = "cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
      load_data(fileName, pre_expand_list[i], party.getSizePreExpand());
    }
    // Create ciphertexts and self-expand
    for (int i = 0; i < NUM_BIT; i++) {
      bool msg = (plaintext >> i) & 1;
      void *main_cipher = std::malloc(party.getSizeMainCipher());
      void *random = std::malloc(party.getSizeRandom());
      void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
      party.encrypt(msg, main_cipher, nullptr, random);
      auto cipher = party.expandWithPlainRandom(pre_expand_list, nullptr,
                                                idParty, main_cipher, random);
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
  }
  // Wait for all parties to create and expand ciphertexts
  std::cout << "INFO: Wait for all parties to create and expand ciphertexts"
            << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < NUM_BIT) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  // During Eval phase
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
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
    maxAll(cipher_list, out_cipher_list, party);
    // Free all input cipher
    for (int i = 0; i < numParty; i++) {
      for (int j = 0; j < NUM_BIT; j++) {
        if (cipher_list[i][j]) {
          delete cipher_list[i][j];
          cipher_list[i][j] = nullptr;
        }
      }
    }
    // Export output cipher
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      party.exportReducedCipher(out_cipher_list[i], output_cipher_mem);
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      save_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      std::free(output_cipher_mem);
    }
    // Free output cipher
    for (int i = 0; i < numParty; i++) {
      if (out_cipher_list[i]) {
        delete out_cipher_list[i];
        out_cipher_list[i] = nullptr;
      }
    }
  }
  // After Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; i < numParty; i++) {
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      fileName += std::to_string(idParty - 1);
      if (!check_data(fileName)) {
        brk = false;
        break;
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import public key
    {
      void *mem = std::malloc(party.getSizePubkey());
      std::string fileNameFormat = "pubkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePubkey());
      party.importPubkey(mem);
      std::free(mem);
    }
    // Import private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      std::string fileNameFormat = "privkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      party.importPrivkey(mem);
      std::free(mem);
    }
    // Import output cipher
    std::vector<thesis::TrlweCipher *> out_cipher_list(numParty, nullptr);
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      load_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      out_cipher_list[i] = party.importReducedCipher(output_cipher_mem);
      std::free(output_cipher_mem);
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
  }
  // Wait for all parties to part decrypt
  std::cout << "INFO: Wait for all parties to part decrypt" << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < numParty) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "output_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Final decrypt and print result
    for (int i = 0; i < numParty; i++) {
      std::vector<thesis::TorusInteger> out_list(numParty);
      std::string fileName;
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      for (int j = 0; j < numParty; j++)
        load_data(fileName + std::to_string(j), &out_list[j],
                  sizeof(out_list[j]));
      double debugError = 0;
      bool msg = party.finDec(out_list.data(), &debugError);
      std::cout << "INFO: Does party " << i << " have the max value? ";
      if (msg)
        std::cout << "Yes" << std::endl;
      else
        std::cout << "No" << std::endl;
      std::cout << "Debug: " << debugError << std::endl;
    }
  }
  return 0;
}
#elif defined(NON_PACKING_EXPAND)
// Max circuit
void maxPairParty(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
                  std::vector<thesis::TrlweCipher *> &out,
                  MpcApplication &party, int partyA, int partyB) {
  DECLARE_TIMING(Debug);
  START_TIMING(Debug);
  {
    auto notA = party.notOp(inp[partyA][NUM_BIT - 1]);
    auto BAndNotA = party.mulOp(inp[partyB][NUM_BIT - 1], notA);
    auto notBOrA = party.notOp(BAndNotA);
    if (out[partyA]) {
      auto temp = out[partyA];
      out[partyA] = party.mulOp(temp, notBOrA);
      delete temp;
    } else
      out[partyA] = party.reduce(notBOrA);
    delete notA;
    delete BAndNotA;
    delete notBOrA;
  }
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << partyA << " " << partyB << std::endl;
  std::cerr << "Debug: " << out[partyA]->_sdError << " "
            << std::sqrt(out[partyA]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  START_TIMING(Debug);
  {
    auto notB = party.notOp(inp[partyB][NUM_BIT - 1]);
    auto AAndNotB = party.mulOp(inp[partyA][NUM_BIT - 1], notB);
    auto notAOrB = party.notOp(AAndNotB);
    if (out[partyB]) {
      auto temp = out[partyB];
      out[partyB] = party.mulOp(temp, notAOrB);
      delete temp;
    } else
      out[partyB] = party.reduce(notAOrB);
    delete notB;
    delete AAndNotB;
    delete notAOrB;
  }
  STOP_TIMING(Debug);
  std::cerr << "Debug: " << partyA << " " << partyB << std::endl;
  std::cerr << "Debug: " << out[partyB]->_sdError << " "
            << std::sqrt(out[partyB]->_varError) << std::endl;
  PRINT_TIMING(Debug);
  thesis::TrgswCipher *Z = nullptr;
  for (int i = NUM_BIT - 1; i >= 1; i--) {
    START_TIMING(Debug);
    auto notAxorB = party.notXorOp(inp[partyA][i], inp[partyB][i]);
    if (Z) {
      auto temp = Z;
      Z = party.mulOp(temp, notAxorB);
      delete temp;
      temp = Z;
      Z = party.mulOp(temp, notAxorB);
      delete temp;
    } else
      Z = party.mulOp(notAxorB, notAxorB);
    delete notAxorB;
    STOP_TIMING(Debug);
    std::cerr << "Debug: " << partyA << " " << partyB << " " << i << std::endl;
    PRINT_TIMING(Debug);
    START_TIMING(Debug);
    {
      auto X = party.mulOp(Z, inp[partyB][i - 1]);
      auto temp = X;
      auto notA = party.notOp(inp[partyA][i - 1]);
      X = party.mulOp(temp, notA);
      delete temp;
      delete notA;
      temp = X;
      X = party.notOp(temp);
      delete temp;
      auto trlwe_temp = out[partyA];
      out[partyA] = party.mulOp(trlwe_temp, X);
      delete X;
      delete trlwe_temp;
    }
    STOP_TIMING(Debug);
    std::cerr << "Debug: " << partyA << " " << partyB << " " << i << std::endl;
    std::cerr << "Debug: " << out[partyA]->_sdError << " "
              << std::sqrt(out[partyA]->_varError) << std::endl;
    PRINT_TIMING(Debug);
    START_TIMING(Debug);
    {
      auto X = party.mulOp(Z, inp[partyA][i - 1]);
      auto temp = X;
      auto notB = party.notOp(inp[partyB][i - 1]);
      X = party.mulOp(temp, notB);
      delete temp;
      delete notB;
      temp = X;
      X = party.notOp(temp);
      delete temp;
      auto trlwe_temp = out[partyB];
      out[partyB] = party.mulOp(trlwe_temp, X);
      delete X;
      delete trlwe_temp;
    }
    STOP_TIMING(Debug);
    std::cerr << "Debug: " << partyA << " " << partyB << " " << i << std::endl;
    std::cerr << "Debug: " << out[partyB]->_sdError << " "
              << std::sqrt(out[partyB]->_varError) << std::endl;
    PRINT_TIMING(Debug);
  }
  delete Z;
  std::cerr << "Debug: " << partyA << " " << partyB << std::endl;
}
void maxAll(std::vector<std::vector<thesis::TrgswCipher *>> &inp,
            std::vector<thesis::TrlweCipher *> &out, MpcApplication &party) {
  DECLARE_TIMING(Full);
  START_TIMING(Full);
  for (size_t i = 0; i < out.size(); i++) {
    for (size_t j = i + 1; j < out.size(); j++)
      maxPairParty(inp, out, party, i, j);
  }
  STOP_TIMING(Full);
  PRINT_TIMING(Full);
}

// Main
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
  int l = PARAM_L;
  double sdFresh = ALPHA;
  // Set id_party and plaintext
  int idParty = -1;
  int plaintext = -1;
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
  if (idParty < 0 || idParty >= numParty || plaintext < 0) {
    std::cerr << "ERROR: id_party is out of range" << std::endl;
    return 1;
  }
  std::cout << "INFO: Your plaintext is " << plaintext << std::endl;
  // Before Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    std::string fileNameFormat = "pubkey_";
    if (check_data(fileNameFormat + std::to_string(idParty - 1)))
      break;
  }
  {
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
    // Export private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      party.exportPrivkey(mem);
      std::string fileNameFormat = "privkey_";
      save_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      std::free(mem);
    }
    // Wait for all parties to broadcast their public keys
    std::cout << "INFO: Wait for all parties to broadcast their public keys"
              << std::endl;
    while (true) {
      bool brk = true;
      std::string fileNameFormat = "pubkey_";
      for (int i = 0; i < numParty; i++) {
        if (!check_data(fileNameFormat + std::to_string(i))) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < numParty; i++) {
        if (i == idParty - 1)
          continue;
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
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
    while (true) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; (i < numParty) && brk; i++) {
        for (int j = 0; j < numParty; j++) {
          if (i == j)
            continue;
          fileName = "pre_expand_";
          fileName += std::to_string(i) + "_";
          fileName += std::to_string(j);
          if (!check_data(fileName)) {
            brk = false;
            break;
          }
        }
      }
      if (brk)
        break;
    }
    // Wait for previous parties
    while (idParty > 0) {
      bool brk = true;
      std::string fileName;
      for (int i = 0; i < NUM_BIT; i++) {
        fileName = "main_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
        fileName = "rand_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(idParty - 1);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
      if (brk)
        break;
    }
    // Create ciphertexts for msg and rand
    for (int i = 0; i < NUM_BIT; i++) {
      bool msg = (plaintext >> i) & 1;
      void *main_cipher = std::malloc(party.getSizeMainCipher());
      void *rand_cipher = std::malloc(party.getSizeRandCipher());
      party.encrypt(msg, main_cipher, rand_cipher, nullptr);
      std::string fileName;
      fileName = "main_cipher_";
      fileName += std::to_string(i) + "_";
      fileName += argv[1];
      save_data(fileName, main_cipher, party.getSizeMainCipher());
      fileName = "rand_cipher_";
      fileName += std::to_string(i) + "_";
      fileName += argv[1];
      save_data(fileName, rand_cipher, party.getSizeRandCipher());
      std::free(main_cipher);
      std::free(rand_cipher);
    }
  }
  // Wait for all parties to create and expand ciphertexts
  std::cout
      << "INFO: Wait for all parties to create ciphertexts for msg and rand"
      << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < NUM_BIT) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "main_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
        fileName = "rand_cipher_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  // Expand ciphertexts
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import all main_cipher, rand_cipher --> Expand --> Export expanded
    // ciphertexts
    void *main_cipher = std::malloc(party.getSizeMainCipher());
    void *rand_cipher = std::malloc(party.getSizeRandCipher());
    void *cipher_mem = std::malloc(party.getSizeExpandedCipher());
    for (int i = 0; i < numParty; i++) {
      std::string fileName;
      std::vector<void *> pre_expand_list(numParty);
      for (int j = 0; j < numParty; j++) {
        if (j == i) {
          pre_expand_list[j] = nullptr;
          continue;
        }
        pre_expand_list[j] = std::malloc(party.getSizePreExpand());
        fileName = "pre_expand_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        load_data(fileName, pre_expand_list[j], party.getSizePreExpand());
      }
      for (int j = 0; j < NUM_BIT; j++) {
        fileName = "main_cipher_";
        fileName += std::to_string(j) + "_";
        fileName += std::to_string(i);
        load_data(fileName, main_cipher, party.getSizeMainCipher());
        fileName = "rand_cipher_";
        fileName += std::to_string(j) + "_";
        fileName += std::to_string(i);
        load_data(fileName, rand_cipher, party.getSizeRandCipher());
        thesis::TrgswCipher *cipher =
            party.expand(pre_expand_list, nullptr, i, main_cipher, rand_cipher);
        party.exportExpandedCipher(cipher, cipher_mem);
        fileName = "cipher_";
        fileName += std::to_string(j) + "_";
        fileName += std::to_string(i);
        save_data(fileName, cipher_mem, party.getSizeExpandedCipher());
      }
      for (int j = 0; j < numParty; j++) {
        if (pre_expand_list[j]) {
          std::free(pre_expand_list[j]);
          pre_expand_list[j] = nullptr;
        }
      }
    }
    std::free(main_cipher);
    std::free(rand_cipher);
    std::free(cipher_mem);
  }
  // During Eval phase
  if (idParty == 0) {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
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
    maxAll(cipher_list, out_cipher_list, party);
    // Free all input cipher
    for (int i = 0; i < numParty; i++) {
      for (int j = 0; j < NUM_BIT; j++) {
        if (cipher_list[i][j]) {
          delete cipher_list[i][j];
          cipher_list[i][j] = nullptr;
        }
      }
    }
    // Export output cipher
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      party.exportReducedCipher(out_cipher_list[i], output_cipher_mem);
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      save_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      std::free(output_cipher_mem);
    }
    // Free output cipher
    for (int i = 0; i < numParty; i++) {
      if (out_cipher_list[i]) {
        delete out_cipher_list[i];
        out_cipher_list[i] = nullptr;
      }
    }
  }
  // After Eval phase
  // Wait for previous parties
  while (idParty > 0) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; i < numParty; i++) {
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      fileName += std::to_string(idParty - 1);
      if (!check_data(fileName)) {
        brk = false;
        break;
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Import public key
    {
      void *mem = std::malloc(party.getSizePubkey());
      std::string fileNameFormat = "pubkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePubkey());
      party.importPubkey(mem);
      std::free(mem);
    }
    // Import private key (Assume that private key is not shared)
    {
      void *mem = std::malloc(party.getSizePrivkey());
      std::string fileNameFormat = "privkey_";
      load_data(fileNameFormat + argv[1], mem, party.getSizePrivkey());
      party.importPrivkey(mem);
      std::free(mem);
    }
    // Import output cipher
    std::vector<thesis::TrlweCipher *> out_cipher_list(numParty, nullptr);
    for (int i = 0; i < numParty; i++) {
      void *output_cipher_mem = std::malloc(party.getSizeReducedCipher());
      std::string fileName;
      fileName = "output_cipher_";
      fileName += std::to_string(i);
      load_data(fileName, output_cipher_mem, party.getSizeReducedCipher());
      out_cipher_list[i] = party.importReducedCipher(output_cipher_mem);
      std::free(output_cipher_mem);
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
  }
  // Wait for all parties to part decrypt
  std::cout << "INFO: Wait for all parties to part decrypt" << std::endl;
  while (true) {
    bool brk = true;
    std::string fileName;
    for (int i = 0; (i < numParty) && brk; i++) {
      for (int j = 0; j < numParty; j++) {
        fileName = "output_";
        fileName += std::to_string(i) + "_";
        fileName += std::to_string(j);
        if (!check_data(fileName)) {
          brk = false;
          break;
        }
      }
    }
    if (brk)
      break;
  }
  {
    // Init party
    MpcApplication party(numParty, idParty, N, m, l, sdFresh);
    // Final decrypt and print result
    for (int i = 0; i < numParty; i++) {
      std::vector<thesis::TorusInteger> out_list(numParty);
      std::string fileName;
      fileName = "output_";
      fileName += std::to_string(i) + "_";
      for (int j = 0; j < numParty; j++)
        load_data(fileName + std::to_string(j), &out_list[j],
                  sizeof(out_list[j]));
      double debugError = 0;
      bool msg = party.finDec(out_list.data(), &debugError);
      std::cout << "INFO: Does party " << i << " have the max value? ";
      if (msg)
        std::cout << "Yes" << std::endl;
      else
        std::cout << "No" << std::endl;
      std::cout << "Debug: " << debugError << std::endl;
    }
  }
  return 0;
}
#else
int main() {
  std::cerr << "ERROR: No specific mode" << std::endl;
  return 1;
}
#endif
