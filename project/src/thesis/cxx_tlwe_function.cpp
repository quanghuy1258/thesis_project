#include "thesis/random.h"
#include "thesis/stream.h"
#include "thesis/tlwe_cipher.h"
#include "thesis/tlwe_function.h"

namespace thesis {

void TlweFunction::genkey(TorusInteger *s, int n, void *streamPtr) {
  if (!s || n < 1)
    return;
  Random::setUniform(s, n, streamPtr);
#ifdef USING_CUDA
  cudaGenkey(s, n, streamPtr);
#else
  auto fn = [s, n]() {
    for (int i = 0; i < n; i++)
      s[i] &= 1;
  };
  if (streamPtr)
    Stream::scheduleS(streamPtr, std::move(fn));
  else
    fn();
#endif
}
void TlweFunction::encrypt(TorusInteger *s, TorusInteger plain,
                           TlweCipher *cipher, void *streamPtr) {
  if (!s || !cipher)
    return;
  Random::setUniform(cipher->_data, cipher->_n, streamPtr);
  Random::setNormalTorus(cipher->_data + cipher->_n, 1, cipher->_sdError,
                         streamPtr);
#ifdef USING_CUDA
  cudaEncrypt(s, plain, cipher, streamPtr);
#else
  auto fn = [s, plain, cipher]() {
    for (int i = 0; i < cipher->_n; i++)
      cipher->_data[cipher->_n] += s[i] * cipher->_data[i];
    TorusInteger bit = 1;
    bit <<= 8 * sizeof(TorusInteger) - 1;
    cipher->_data[cipher->_n] += plain * bit;
  };
  if (streamPtr)
    Stream::scheduleS(streamPtr, std::move(fn));
  else
    fn();
#endif
}
void TlweFunction::decrypt(TorusInteger *s, TlweCipher *cipher,
                           TorusInteger *plain, double *abs_err,
                           void *streamPtr) {
  if (!s || !cipher || (!plain && !abs_err))
    return;
#ifdef USING_CUDA
  cudaDecrypt(s, cipher, plain, abs_err, streamPtr);
#else
  auto fn = [s, cipher, plain, abs_err]() {
    TorusInteger x = cipher->_data[cipher->_n];
    for (int i = 0; i < cipher->_n; i++)
      x -= s[i] * cipher->_data[i];
    double y = std::abs(x / std::pow(2, 8 * sizeof(TorusInteger)));
    if (plain)
      *plain = (y < 0.25) ? 0 : 1;
    if (abs_err)
      *abs_err = (y < 0.25) ? y : (0.5 - y);
  };
  if (streamPtr)
    Stream::scheduleS(streamPtr, std::move(fn));
  else
    fn();
#endif
}

} // namespace thesis
