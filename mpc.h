#ifndef _MPC_
#define _MPC_
#include "thesis/declarations.h"
#include "thesis/load_lib.h"
#include "thesis/trgsw.h"
#include "thesis/random.h"
#include "thesis/helper.h"
namespace thesis {
    
    class Party
    {

    private:
        int _value;

        bool isProcessingValue;
        bool isProcessingKey;
        PolynomialBinary _privateKey;
        std::vector<PolynomialTorus> _publicKey;
        // _listPublicKey contain self public key
        // party[numberOfParty].PublicKey
        std::vector<std::vector<PolynomialTorus>> _listPublicKey;

        // bit[linput].party[numberOfParty].ciphertext
        std::vector<std::vector<std::vector<PolynomialTorus>>> _listCipher;

        // The bonus ciphers are list cipher of each entry of Random
        // bit[linput].party[numberOfParty].entryOfRandomnessOfCiphertext[(k+1)*l*M].ciphertext
        std::vector<std::vector<std::vector<std::vector<PolynomialTorus>>>> _listBonusCipher;

        // bit[linput].party[numberOfParty].entryOfExpandCipher[numberOfParty^2].ciphertext
        std::vector<std::vector<std::vector<std::vector<PolynomialTorus>>>> _listExpandCipher;
        
        // bit[linput].matrixBit
        std::vector<std::vector<bool>> _listRandom;

        // srcParty[numberOfParty].dstParty[numberOfParty].extendPrivateKey
        std::vector<std::vector<std::vector<PolynomialTorus>>> _listExtendPrivateKeyOfOther;
        
        int selfIndex;
        int numberOfParty;

        /**
        * @brief _encrypt is a private method for encrypt ID-th bit of the Party's value
        * 
        * @param ID 
        */
        void _encrypt(int ID);

        std::vector<PolynomialTorus> _extend(int bit, int srcParty, int dstParty);

        void _bitDecomp(const std::vector<PolynomialTorus> &inp, std::vector<PolynomialTorus> &out, int wInp);

        void _encrypt(std::vector<PolynomialTorus> &cipher, std::vector<bool> &randomness, const bool &plain);

        void _getSelfIndex();

        /**
         * @brief 
         * 
         * @param ID 
         */
        void _expand(int bit, int srcParty);

        /**
        * @brief 
        * 
        * @return std::vector<PolynomialTorus> 
        */
        std::vector<PolynomialTorus> _generatePublicKey();

    public:

        /**
        * @brief Construct a new Party object
        * 
        * Creating a party with random params
        */
        Party();

        /**
        * @brief Construct a new Party object
        * 
        * Creating the party with user's private key
        * @param privateKey The private key of user - Binary polynomial
        */
        Party(PolynomialBinary privateKey);

        /**
        * @brief KeyGen generate keypair and replace current keypair of the party.
        *  This function can not be called if the current key pair is using for a protocol.
        * 
        * @return true Created
        * @return false Can not create key pair
        */
        bool KeyGen();

        /**
        * @brief 
        * 
        * @param ID Index of publicKey. ID = -1: generate extend privateKey for all of publicKeys. ID > -1: for ID-th publicKey.
        * @return true Generated.
        * @return false Can not generate extend privateKey.
        */
        bool GenExtendPrivateKey(int ID);

        /**
        * @brief SetNewValue will replace current value of party if the value is not be processing
        * 
        * @return true Replaced
        * @return false Can not replace
        */
        bool SetNewValue(int);

        /**
        * @brief Encrypt function: Encrypting the ID-th bit of value or all of bits
        * 
        * @param ID -1: Encrypting all of bits; > -1: Encrypting the ID-th bif of value;
        * @return true Encrypted
        * @return false Can not encrypt because some error
        */
        bool Encrypt(int ID = -1);

        /**
         * @brief 
         * 
         * @param ID 
         * @return true 
         * @return false 
         */
        bool Expand(int bit = -1, int partyID = -1);

        /**
        * @brief AddPeer Add publickey of party in protocol
        * 
        * @param peerPublicKey 
        * @return true 
        * @return false 
        */
        bool AddPeer(std::vector<PolynomialTorus> peerPublicKey);

        bool AddCipherTextOfBitOfParty(int bit, int ID, const std::vector<PolynomialTorus> cipherOfParty);
        bool AddBonusCipherOfBitOfParty(int bit, int ID, const std::vector<std::vector<PolynomialTorus>> BonusCipherOfParty);

        void Eval();

        /**
        * @brief GetExtendPrivateKey get extendPrivateKey for ID-th publicKey in list publicKeys.
        * ExtendPrivateKey of Party[i] for Party[j] is (Pk[i]-Pk[j])*[-sk[i], 1] + error = b[i] - b[j] - a[i](*)sk[i] + a[j](*)sk[i]
        * 
        * @param ID 
        * @return std::vector<PolynomialTorus>* 
        */
        const std::vector<PolynomialTorus> &GetExtendPrivateKey(int srcParty, int dstParty) const;
        const std::vector<PolynomialTorus> &GetPublicKey() const;

        /**
        * @brief Destroy the Party:: Party object
        * 
        */
        ~Party();

        //size of std::vector<PolynomialTorus> is (k+1)*M
        static int M;

        //size of std::vector<PolynomialTorus> is (k+1)*M
        static int k;

        //Torus[X^N + 1]
        static int N;

        static int l;

        //value must be an integer lInput bits.
        static int lInput;
    };

}
#endif