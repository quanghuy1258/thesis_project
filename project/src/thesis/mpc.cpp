#include "thesis/mpc.h"
namespace thesis {

    void Party::_encrypt(int bit)
    {
        this->_encrypt(this->_listCipher[bit][this->selfIndex], this->_listRandom[bit], (this->_value>>bit) & 1);
        this->_listBonusCipher[bit].resize((k+1)*l*M);
        std::vector<bool> randomTemp;
        for (int i = 0; i < (k+1)*l; ++i) {
            for (int j = 0; j < M; ++j) {
                this->_encrypt(this->_listBonusCipher[bit][this->selfIndex][i*M + j], randomTemp, this->_listRandom[bit][i*M + j]);
            }
        }
        //TODO handle error
    }

    void Party::_encrypt(std::vector<PolynomialTorus> &cipher, std::vector<bool> &randomness, const bool &plain)
    {
        randomness.resize((k+1)*l*M);
        for (auto &&entry : randomness)
        {
            entry = Random::getUniformInteger() & 1;
        }
        cipher.resize((k+1)*l*(k+1));
        //cipher = R(*)pk
        bool nonError = BMatrixPolyMul(cipher, this->_publicKey, M, randomness, (k+1)*l);
        if (nonError) {
            if (plain) {
                nonError = BAddGadget(cipher, (k+1)*l);
            }
        }
        //TODO handle error
    }

    std::vector<PolynomialTorus> Party::_extend(int bit, int srcParty, int dstParty)
    {
        std::vector<PolynomialTorus> res;
        std::vector<PolynomialTorus> bitDecompResult;
        res.resize((k+1)*l*(k+1));
        std::vector<std::vector<PolynomialTorus>> Z;
        Z.resize((k+1)*l*M);
        for (int i=0; i<(k+1)*l; ++i) {
            for (int j=0; j<M; ++j) {
                Z[i*M + j].resize((k+1)*l*(k+1));
                memset(&Z[i*M + j], 0, (k+1)*l*(k+1)*N*sizeof(Torus));
                Z[i*M + j][k*(k+1) + j] = this->_listExtendPrivateKeyOfOther[dstParty][srcParty][j];
            }
        }
        memset(&res, 0, (k+1)*l*(k+1)*N*sizeof(Torus));
        bool err;
        for (int i=0; i<(k+1)*l*M; ++i) {
            this->_bitDecomp(Z[i], bitDecompResult, (k+1)*l);
            err = BMatrixPolyAdd(res, res, OMatrixPolyMul(bitDecompResult, (k+1)*l, this->_listBonusCipher[bit][srcParty][i], (k+1)*l));
        }
        return res;
    }

    void Party::_expand(int bit, int srcParty)
    {
        this->_listExpandCipher[bit][srcParty].resize(this->numberOfParty*this->numberOfParty);

        for (int i = 0; i < this->numberOfParty; ++i) {
            for (int j = 0; j < this->numberOfParty; ++j) {
                if (i==j) {
                    this->_listExpandCipher[bit][srcParty][i*this->numberOfParty + j] = this->_listCipher[bit][srcParty];
                } else {
                    if (j==srcParty) {
                        this->_listExpandCipher[bit][srcParty][i*this->numberOfParty + j] = this->_extend(bit,srcParty,j);
                    } else {
                        this->_listExpandCipher[bit][srcParty][i*this->numberOfParty + j].resize((k+1)*l*(k+1));
                        memset(&this->_listExpandCipher[bit][srcParty][i*this->numberOfParty + j], 0, (k+1)*l*(k+1)*N*sizeof(Torus));
                    }
                    
                }
            }
        }
    }

    std::vector<PolynomialTorus> Party::_generatePublicKey()
    {
        //Public key is [a, b], a, b are vector of polynomial over Torus[X^N + 1], size of a, b is M, b = a(*)s + e; s is privateKey; 
        this->_publicKey.resize(M*(k+1));
        for (int i = 0; i < M; ++i){
            this->_publicKey[i].resize(N);
            this->_publicKey[i + k*M].resize(N);
            //a = uniform random polynomial torus
            //b = sample an error polynomial torus 
            for (int j=0; j<N; ++j) {
                this->_publicKey[i][j] = Random::getUniformTorus();
                this->_publicKey[i+k*M][j] = Random::getNormalTorus(0, STDDEV_ERROR);
            }

            //b = b + a*s
            this->_publicKey[i + k*M] = PolyAdd(this->_publicKey[i + k*M], PolyMul(this->_publicKey[i], this->_privateKey));
        }
    }
    
    Party::Party()
    {
        this->KeyGen();
        this->isProcessingKey = false;
        this->isProcessingValue = false;
        this->_value = 0;
    }
    
    Party::Party(PolynomialBinary privateKey)
    {
        this->_privateKey = privateKey;
        this->_publicKey = this->_generatePublicKey();
    }

    bool Party::KeyGen()
    {
        if (this->isProcessingKey) return false;
        this->_privateKey.resize(N);
        for (auto &&entry : this->_privateKey)
        {   
            entry = Random::getUniformInteger()&1;
        }
        this->_generatePublicKey();
        return true;
    }

    bool Party::SetNewValue(int value)
    {
        if (this->isProcessingValue) return false;
        this->_value = value;
        return true;
    }

    bool Party::AddPeer(std::vector<PolynomialTorus> peerPublicKey)
    {
        //TODO Check correctness of publickey
        if (this->isProcessingKey) return false;
        this->_listPublicKey.push_back(peerPublicKey);
        if (this->_listPublicKey.size() + 1 == this->numberOfParty) {
            this->_getSelfIndex();
            this->isProcessingKey = true;
        }
        return true;
    }

    bool Party::GenExtendPrivateKey(int dstParty)
    {
        if ((dstParty >= this->_listPublicKey.size()) && (dstParty < -1)) {
            return false; 
        }
        int startIndex = 0;
        int endIndex = this->_listPublicKey.size() - 1;
        if (dstParty != -1) {
            startIndex = dstParty;
            endIndex = dstParty;
        }
        for (int p = startIndex; p <= endIndex; ++p) {
            this->_listExtendPrivateKeyOfOther[this->selfIndex][dstParty].resize(M);
            for (int i = 0; i < M; i++) {
                this->_listExtendPrivateKeyOfOther[this->selfIndex][dstParty][i].resize(N);
                //b[i]-a[i](*)sk[i]
                this->_listExtendPrivateKeyOfOther[this->selfIndex][dstParty][i] = PolySub(this->_publicKey[i + k*M], PolyMul(this->_publicKey[i], this->_privateKey));
                //(b[i]-a[i](*)sk[i]) + (a[j](*)sk[i] - b[j])
                this->_listExtendPrivateKeyOfOther[this->selfIndex][dstParty][i] = PolyAdd(this->_listExtendPrivateKeyOfOther[this->selfIndex][dstParty][i], PolySub(PolyMul(this->_listPublicKey[p][i], this->_privateKey), this->_listPublicKey[p][i + k*M]));
                //((b[i]-a[i](*)sk[i]) + (a[j](*)sk[i] - b[j])) + error
                PolyAddError(this->_listExtendPrivateKeyOfOther[this->selfIndex][dstParty][i]);
            }
        }
        return true;
    }

    const std::vector<PolynomialTorus> &Party::GetExtendPrivateKey(int srcParty, int dstParty) const
    {
        return this->_listExtendPrivateKeyOfOther[srcParty][dstParty];
    }

    const std::vector<PolynomialTorus> &Party::GetPublicKey() const
    {
        return this->_publicKey;
    }

    bool Party::Encrypt(int ID = -1)
    {
        this->isProcessingKey = true;
        this->isProcessingValue = true;
        int startIndex = 0;
        int endIndex = this->_listPublicKey.size() - 1;
        if (ID != -1) {
            startIndex = ID;
            endIndex = ID;
        }
        for (int i=startIndex; i<endIndex; ++i) {
            this->_encrypt(i);
        }
        return true;
    }

    bool Party::Expand(int bit = -1, int partyID = -1)
    {
        int startIndex = 0;
        int endIndex = this->_listPublicKey.size() - 1;
        if (partyID != -1) {
            startIndex = partyID;
            endIndex = partyID;
        }
        int startBit = 0;
        int endBit = this->lInput - 1;
        if (bit != -1) {
            startBit = bit;
            endBit = bit;
        }
        for (int b = startBit; b<endBit; ++b) {
            for (int i=startIndex; i<endIndex; ++i) {
                for (int j = 0; j<this->numberOfParty - 1; ++j)
                    this->_expand(b, i);
            }
        }
    }

    Party::~Party()
    {
    }
    
}