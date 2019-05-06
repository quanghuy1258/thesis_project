#include "thesis/helper.h"

namespace thesis {
    PolynomialTorus PolyMul(const PolynomialTorus &polyA, const PolynomialTorus &polyB) {
    }

    PolynomialTorus PolyMul(const PolynomialTorus &polyA, const PolynomialBinary &polyB) {
    }

    PolynomialTorus PolyAdd(const PolynomialTorus &polyA, const PolynomialTorus &polyB){
        
    }

    PolynomialTorus PolySub(const PolynomialTorus &polyA, const PolynomialTorus &polyB) {

    }

    std::vector<PolynomialTorus> OMatrixPolyMul(const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialTorus> &matrixB, int wB)
    {

    }

    std::vector<PolynomialTorus> OMatrixPolyMul(const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialBinary> &matrixB, int wB)
    {

    }

    std::vector<PolynomialTorus> OMatrixPolyMul(const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<std::vector<bool>> &matrixB, int wB)
    {

    }

    std::vector<PolynomialTorus> OAddGadget(const std::vector<PolynomialTorus> &matrixIn, int wA)
    {
        
    }

    void PolyAddError(PolynomialTorus &polyA) {
        for (auto &&entry : polyA)
        {
            entry += Random::getNormalTorus(0, STDDEV_ERROR);
        }
    }

    bool BMatrixPolyMul(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialTorus> &matrixB, int wB)
    {

    }

    bool BMatrixPolyMul(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialBinary> &matrixB, int wB)
    {

    }

    bool BMatrixPolyMul(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<bool> &matrixB, int wB)
    {

    }


    bool BMatrixPolyAdd(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, const std::vector<PolynomialTorus> &matrixB)
    {

    }

    bool BAddGadget(std::vector<PolynomialTorus> &matrixOut, const std::vector<PolynomialTorus> &matrixIn, int wA)
    {

    }

    bool BAddGadget(std::vector<PolynomialTorus> &matrix, int wA)
    {

    }
}