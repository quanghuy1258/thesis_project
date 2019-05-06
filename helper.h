#ifndef _HELPER_
#define _HELPER_
#include "thesis/trgsw.h"
#include "thesis/load_lib.h"
#include "thesis/declarations.h"
#include "thesis/random.h"
namespace thesis
{
    /**
     * @brief PolyMul - function for polynomial multiplication
     * 
     * @param polyA 
     * @param polyB 
     * @return PolynomialTorus = polyA (*) polyB
     */
    PolynomialTorus PolyMul(const PolynomialTorus &polyA, const PolynomialTorus &polyB);
    PolynomialTorus PolyMul(const PolynomialTorus &polyA, const PolynomialBinary &polyB);

    /**
     * @brief PolyAdd - 
     * 
     * @param polyA 
     * @param polyB 
     * @return PolynomialTorus = polyA (+) polyB
     */
    PolynomialTorus PolyAdd(const PolynomialTorus &polyA, const PolynomialTorus &polyB);

    /**
     * @brief PolySub - 
     * 
     * @param polyA 
     * @param polyB 
     * @return PolynomialTorus = polyA (-) polyB
     */
    PolynomialTorus PolySub(const PolynomialTorus &polyA, const PolynomialTorus &polyB);
    
    /**
     * @brief 
     * 
     * @param polyA 
     */
    void PolyAddError(PolynomialTorus &polyA);

    /**
     * @brief OMatrixPolyMul Multiple two matrix of polynomial and return an Object Matrix PolynomialTorus
     * 
     * @param matrixA 
     * @param wA 
     * @param matrixB 
     * @param wB 
     * @return std::vector<PolynomialTorus> 
     */
    std::vector<PolynomialTorus> OMatrixPolyMul(const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialTorus> &matrixB, int wB);
    std::vector<PolynomialTorus> OMatrixPolyMul(const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialBinary> &matrixB, int wB);
    std::vector<PolynomialTorus> OMatrixPolyMul(const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<std::vector<bool>> &matrixB, int wB);

    std::vector<PolynomialTorus> OAddGadget(const std::vector<PolynomialTorus> &matrixIn, int wA);

    /**
     * @brief Multiple two matrix of polynomial and return boolean
     * 
     * @param matrixC 
     * @param matrixA 
     * @param wA 
     * @param matrixB 
     * @param wB 
     * @return true 
     * @return false 
     */
    bool BMatrixPolyMul(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialTorus> &matrixB, int wB);
    bool BMatrixPolyMul(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<PolynomialBinary> &matrixB, int wB);
    bool BMatrixPolyMul(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, int wA, const std::vector<bool> &matrixB, int wB);


    bool BMatrixPolyAdd(std::vector<PolynomialTorus> &matrixC, const std::vector<PolynomialTorus> &matrixA, const std::vector<PolynomialTorus> &matrixB);

    /**
     * @brief 
     * 
     * @param matrixOut 
     * @param matrixIn 
     * @param wA 
     * @return true 
     * @return false 
     */
    bool BAddGadget(std::vector<PolynomialTorus> &matrixOut, const std::vector<PolynomialTorus> &matrixIn, int wA);
    bool BAddGadget(std::vector<PolynomialTorus> &matrix, int wA);

} // namespace thesis
#endif