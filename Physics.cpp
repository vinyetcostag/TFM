#include <hl_GlobalBasisFunctions.h>
#include <hl_HiPerProblem.h>

#include "Physics.h"


void ReactionDiffusion(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["morphogens"];
    int pDim = subFill.pDim;                         // dimension of the parameterized object

    int numDOFs = subFill.numDOFs;
    int eNN  = subFill.eNN;

    wrapper<double, 1> bf(subFill.nborBFs(), eNN);
    wrapper<double, 2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);
    wrapper<double, 2> nborDOFs0(subFill.nborDOFs0.data(), eNN, numDOFs);

    double jac{};
    tensor<double, 2> Dbfdx(eNN, pDim);
    GlobalBasisFunctions::gradients(Dbfdx, jac, subFill);

    using ttl::index::N, ttl::index::M;
    using ttl::index::I, ttl::index::J;
    using ttl::index::a, ttl::index::b, ttl::index::c, ttl::index::d;
    using ttl::index::i;

    tensor<double, 1> mg_tN = nborDOFs0(N, a) * bf(N);
    tensor<double, 1> mg_tN1 = nborDOFs(N, a) * bf(N);
    tensor<double, 2> dmgdx = Dbfdx(N, a) * nborDOFs(N, i);

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, numDOFs, eNN, numDOFs);
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, numDOFs);

    const double dt  = fillStr.getRealParameter(Params::dt);
    const double dc  = fillStr.getRealParameter(Params::dc);
    const double dh  = fillStr.getRealParameter(Params::dh);
    const double rhoc = fillStr.getRealParameter(Params::rhoc);
    const double rhoh = fillStr.getRealParameter(Params::rhoh);

    const double cN = mg_tN(0);
    const double cN1 = mg_tN1(0);
    const double hN = mg_tN(1);
    const double hN1 = mg_tN1(1);

    Bk(I, 0) = jac*(bf(I) * (cN1 - cN) / dt
            + dc * dmgdx(a, 0) * Dbfdx(I, a)
            - rhoc * bf(I) * (cN1 * cN1 / hN1 - cN1) );
    Bk(I, 1) = jac*(bf(I) * (hN1 - hN) / dt
            + dh * dmgdx(a, 1) * Dbfdx(I, a)
            - rhoh * bf(I) *(cN1 * cN1 - hN1));

    Ak(I, 0, J, 0) = jac * ( bf(I) * bf(J) / dt
            + dc * Dbfdx(I, a) * Dbfdx(J, a)
            - rhoc * bf(I) * bf(J) * (2.0 * cN1 / hN1 - 1.0) );
    Ak(I, 0, J, 1) = (jac * rhoc * cN1 * cN1 / hN1 / hN1) * bf(I) * bf(J);

    Ak(I, 1, J, 1) = jac * ( bf(I) * bf(J) / dt
                             + dh * Dbfdx(I, a) * Dbfdx(J, a)
                             + rhoh * bf(I) * bf(J) );
    Ak(I, 1, J, 0) = -(jac * rhoh * 2.0 * cN1 ) * bf(I) * bf(J);
}

void ConvectionDiffusion(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["morphogens"];
    int pDim = subFill.pDim;                         // dimension of the parametrized object

    int numDOFs = subFill.numDOFs;
    int eNN = subFill.eNN;

    wrapper<double, 1> bf(subFill.nborBFs(), eNN);
    wrapper<double, 2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);
    wrapper<double, 2> nborDOFs0(subFill.nborDOFs0.data(), eNN, numDOFs);
    wrapper<double, 2> nborAux(subFill.nborAuxF.data(), eNN, subFill.numAuxF);

    double jac{};
    tensor<double, 2> Dbfdx(eNN, pDim);
    GlobalBasisFunctions::gradients(Dbfdx, jac, subFill);

    using ttl::index::I, ttl::index::J, ttl::index::N;
    using ttl::index::a, ttl::index::b;

    tensor<double, 1> mg_tN = nborDOFs0(I, a) * bf(I);
    tensor<double, 1> mg_tN1 = nborDOFs(I, a) * bf(I);
    tensor<double, 2> dmgdx = Dbfdx(I,a) * nborDOFs(I,b);

    tensor<double, 2> nborVel = nborAux(ttl::all, ttl::range(0,1));
    tensor<double, 1> vel = nborVel(I,a) * bf(I);

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, numDOFs, eNN, numDOFs);
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, numDOFs);

    const double dt  = fillStr.getRealParameter(Params::dt);
    const double dc  = fillStr.getRealParameter(Params::dc);
    const double dh  = fillStr.getRealParameter(Params::dh);
    const double rhoc = fillStr.getRealParameter(Params::rhoc);
    const double rhoh = fillStr.getRealParameter(Params::rhoh);

    const double cN = mg_tN(0);
    const double cN1 = mg_tN1(0);
    const double hN = mg_tN(1);
    const double hN1 = mg_tN1(1);
    const double mN = mg_tN(2);
    const double mN1 = mg_tN1(2);

    Bk(I, 0) = jac*(bf(I) * (cN1 - cN) / dt 
                     + dc * dmgdx(a, 0) * Dbfdx(I, a)
                     - rhoc * bf(I) * (cN1 * cN1 / hN1 - cN1) );
    Bk(I, 0) += jac * bf(I) * (cN1 * Dbfdx(N, a) * nborVel(N, a) + vel(a) * dmgdx(a, 0)) ;

    Bk(I, 1) = jac*(bf(I) * (hN1 - hN) / dt
                     + dh * dmgdx(a, 1) * Dbfdx(I, a)
                     - rhoh * bf(I) *(cN1 * cN1 - hN1));
    Bk(I, 1) += jac * bf(I) * (hN1 * Dbfdx(N, a) * nborVel(N, a) + vel(a) * dmgdx(a, 1)) ;
    
    Bk(I, 2) = jac*(bf(I) * (mN1 - mN) / dt);
    Bk(I,2) += jac * bf(I) * (mN1 * Dbfdx(N, a) * nborVel(N, a) + vel(a) * dmgdx(a, 2)) ;

    Ak(I, 0, J, 0) = jac * ( bf(I) * bf(J) / dt
                             + dc * Dbfdx(I, a) * Dbfdx(J, a)
                             - rhoc * bf(I) * bf(J) * (2.0 * cN1 / hN1 - 1.0) );
    Ak(I, 0, J, 0) += jac * bf(I) * (bf(J) * Dbfdx(N, a) * nborVel(N, a) + vel(a) * Dbfdx(J, a)) ;

    Ak(I, 0, J, 1) = (jac * rhoc * cN1 * cN1 / hN1 / hN1) * bf(I) * bf(J);

    Ak(I, 1, J, 1) = jac * ( bf(I) * bf(J) / dt
                             + dh * Dbfdx(I, a) * Dbfdx(J, a)
                             + rhoh * bf(I) * bf(J) );
    Ak(I, 1, J, 1) += jac * bf(I) * (bf(J) * Dbfdx(N, a) * nborVel(N, a) + vel(a) * Dbfdx(J, a)) ;
    Ak(I, 1, J, 0) = -(jac * rhoh * 2.0 * cN1 ) * bf(I) * bf(J);
    
    Ak(I, 2, J, 2) = jac * ( bf(I) * bf(J) / dt);
    Ak(I, 2, J, 2) += jac * bf(I) * (bf(J) * Dbfdx(N, a) * nborVel(N, a) + vel(a) * Dbfdx(J, a)) ;
}

void ConvectionDiffusionALE(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["morphogens"];
    int pDim = subFill.pDim;                         // dimension of the parametrizied object

    int numDOFs = subFill.numDOFs;
    int eNN = subFill.eNN;

    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, numDOFs);
    wrapper<double,2> nborAux(subFill.nborAuxF.data(), eNN, subFill.numAuxF);

    double jac{};
    tensor<double, 2> Dbfdx(eNN, pDim);
    GlobalBasisFunctions::gradients(Dbfdx, jac, subFill);

    using ttl::index::N, ttl::index::M;
    using ttl::index::I, ttl::index::J;
    using ttl::index::a, ttl::index::b, ttl::index::c, ttl::index::d;
    using ttl::index::i;

    tensor<double,1> mg_tN = nborDOFs0(N, a) * bf(N);
    tensor<double,1> mg_tN1 = nborDOFs(N, a) * bf(N);
    tensor<double,2> dmgdx = Dbfdx(N, a) * nborDOFs(N, i);

    tensor<double,2> nborVel = nborAux(ttl::all, ttl::range(0,1));
    tensor<double,2> nborUN1 = nborAux(ttl::all, ttl::range(2,3));
    tensor<double,2> nborUN  = nborAux(ttl::all, ttl::range(4,5));
    tensor<double,1> vel = nborVel(N, a) * bf(N);
    tensor<double,1> uN1 = nborUN1(N, a) * bf(N);
    tensor<double,1> uN = nborUN(N, a) * bf(N);

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, numDOFs, eNN, numDOFs);
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, numDOFs);

    const double dt  = fillStr.getRealParameter(Params::dt);
    const double dc  = fillStr.getRealParameter(Params::dc);
    const double dh  = fillStr.getRealParameter(Params::dh);
    const double rhoc = fillStr.getRealParameter(Params::rhoc);
    const double rhoh = fillStr.getRealParameter(Params::rhoh);

    const double cN = mg_tN(0);
    const double cN1 = mg_tN1(0);
    const double hN = mg_tN(1);
    const double hN1 = mg_tN1(1);
    const double mN = mg_tN(2);
    const double mN1 = mg_tN1(2);

    Bk(I, 0) = jac*(bf(I) * (cN1 - cN) / dt
                     + dc * dmgdx(a, 0) * Dbfdx(I, a)
                     - rhoc * bf(I) * (cN1 * cN1 / hN1 - cN1) );
    Bk(I, 0) += jac * bf(I) * (cN1 * Dbfdx(N, a) * nborVel(N, a) + vel(a) * dmgdx(a, 0)) ;
    Bk(I, 0) -= jac / (dt) * bf(I) * (uN1(a) - uN(a)) * dmgdx(a, 0);

    Bk(I, 1) = jac*(bf(I) * (hN1 - hN) / dt
                     + dh * dmgdx(a, 1) * Dbfdx(I, a)
                     - rhoh * bf(I) *(cN1 * cN1 - hN1));
    Bk(I, 1) += jac * bf(I) * (hN1 * Dbfdx(N, a) * nborVel(N, a) + vel(a) * dmgdx(a, 1)) ;
    Bk(I, 1) -= jac / (dt) * bf(I) * (uN1(a) - uN(a)) * dmgdx(a, 1);

    Bk(I, 2) = jac * (bf(I)*(mN1 - mN) / dt
                     + bf(I)*(mN1*Dbfdx(N,a)*nborVel(N,a)+vel(a)*dmgdx(a,2)));
    Bk(I, 2) -= jac / (dt) * bf(I) * (uN1(a) - uN(a)) * dmgdx(a, 2);

    Ak(I, 0, J, 0) = jac * ( bf(I) * bf(J) / dt
                             + dc * Dbfdx(I, a) * Dbfdx(J, a)
                             - rhoc * bf(I) * bf(J) * (2.0 * cN1 / hN1 - 1.0) );
    Ak(I, 0, J, 0) += jac * bf(I) * (bf(J) * Dbfdx(N, a) * nborVel(N, a) + vel(a) * Dbfdx(J, a)) ;
    Ak(I, 0, J, 0) -= jac / (dt) * bf(I) * (uN1(a) - uN(a)) * Dbfdx(J,a) ;

    Ak(I, 0, J, 1) = (jac * rhoc * cN1 * cN1 / hN1 / hN1) * bf(I) * bf(J);

    Ak(I, 1, J, 0) = -(jac * rhoh * 2.0 * cN1 ) * bf(I) * bf(J);

    Ak(I, 1, J, 1) = jac * ( bf(I) * bf(J) / dt
                             + dh * Dbfdx(I, a) * Dbfdx(J, a)
                             + rhoh * bf(I) * bf(J) );
    Ak(I, 1, J, 1) += jac * bf(I) * (bf(J) * Dbfdx(N, a) * nborVel(N, a) + vel(a) * Dbfdx(J, a)) ;
    Ak(I, 1, J, 1) -= jac / (dt) * bf(I) * (uN1(a) - uN(a)) * Dbfdx(J,a) ;

    Ak(I,2,J,2) = jac * (bf(I) *bf(J)/dt + bf(I) * (bf(J)*Dbfdx(N,a)*nborVel(N,a) + vel(a) * Dbfdx(J, a)));
    Ak(I,2,J,2) -= jac/ (dt) *bf(I) * (uN1(a)-uN(a))* Dbfdx(J,a) ;

    fillStr.addToGlobalIntegral("area", jac);
    fillStr.addToGlobalIntegral("mass", jac * mN1);
    // add integration of the mass derivative, should be zero
}

void TensionFlow(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["velocity"];
    int pDim = subFill.pDim;                         // dimension of the geometry
    int DOF = subFill.numDOFs;                       // degrees of freedom
    int eNN  = subFill.eNN;                          // number of nodes of the neighborhood

    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    double jac{};
    tensor<double,2> dbfdx(eNN, pDim);
    GlobalBasisFunctions::gradients(dbfdx, jac, subFill);

    using ttl::index::I, ttl::index::J, ttl::index::N;
    using ttl::index::a, ttl::index::b, ttl::index::c, ttl::index::d;

    const double mu  = fillStr.getRealParameter(Params::mu);
    const double nu  = fillStr.getRealParameter(Params::nu);

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, DOF, eNN, DOF);
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, DOF);

    Ak(I,a,J,b) = jac * mu * ((dbfdx(I, c) * Identity2(a, b) * dbfdx(J, c)) + (dbfdx(I, b) * dbfdx(J, a)) );
    Ak(I,a,J,b) += jac * nu * bf(I) * bf(J) * Identity2(a, b);

    wrapper<double,2> nborAux(subFill.nborAuxF.data(), eNN, subFill.numAuxF);

    double cc = bf(N) * nborAux(N, 0);
    double mm = bf(N) * nborAux(N, 2);
    double gamma = fillStr.getRealParameter(Params::gamma);
    double k = fillStr.getRealParameter(Params::k);
    double m0 = fillStr.getRealParameter(Params::m0);

    double cs = fillStr.getRealParameter(Params::cs);
    //double σ = gamma * (cc/cs)*(cc/cs)/(1.+(cc/cs)*(cc/cs)) + k * (1.0 - (mm / m0))*(1.0 - (mm / m0));
    double σ = (gamma * (cc/cs)*(cc/cs)/(1+(cc/cs)*(cc/cs)) + k * (1.0 - (mm / m0)*(mm / m0)));
    Bk(I,a) = - jac * dbfdx(I, a) * σ;
}

double CheckCFL(hiperlife::SmartPtr<hiperlife::DOFsHandler>& velocity)
{
    using namespace hiperlife;
    using std::vector;

    DistributedMesh& mesh = *velocity->mesh;

    double minDeltaT{10000};
    for(int e = 0; e < mesh.loc_nElem(); e++)
    {
        // FIXME Only useful for Lagrangian elements
        vector<int> elemNodes = mesh.elemNodeNbors(e, IndexType::Local);
        // Compute max velocity in element
        double maxVelNorm{};
        for(const int node : elemNodes) {
            double nodeVelNorm2{};
            nodeVelNorm2 += velocity->nodeDOFs->getValue(0, node, IndexType::Global) * velocity->nodeDOFs->getValue(0, node, IndexType::Global);
            nodeVelNorm2 += velocity->nodeDOFs->getValue(1, node, IndexType::Global) * velocity->nodeDOFs->getValue(1, node, IndexType::Global);
            if(velocity->nodeDOFs->numFlds() >= 3)
                nodeVelNorm2 += velocity->nodeDOFs->getValue(2, node, IndexType::Global) * velocity->nodeDOFs->getValue(2, node, IndexType::Global);

            if(nodeVelNorm2 > maxVelNorm)
                maxVelNorm = nodeVelNorm2;
        }
        maxVelNorm = sqrt(maxVelNorm);

        // Compute min dx in element
        vector<double> elemCoords = mesh.elemNborNodeCoords(e, IndexType::Local);
        int NumNodes = elemCoords.size()/3;
        double minDeltaX{1000000};
        for(int i = 0; i < NumNodes; i++) {
            for(int j = 0; j < i; j++) {
                double* nodeI = &elemCoords[3*i+0];
                double* nodeJ = &elemCoords[3*j+0];
                double DeltaX = (nodeI[0]-nodeJ[0])*(nodeI[0]-nodeJ[0]) +
                        (nodeI[1]-nodeJ[1])*(nodeI[1]-nodeJ[1]) +
                        (nodeI[2]-nodeJ[2])*(nodeI[2]-nodeJ[2]);

                if(DeltaX < minDeltaX)
                    minDeltaX = DeltaX;
            }
        }
        minDeltaX = sqrt(minDeltaX);

        const double minElemDeltaT = minDeltaX / maxVelNorm;
        if(minElemDeltaT < minDeltaT)
            minDeltaT = minElemDeltaT;
    }

    double recDeltaT{};
    MPI_Allreduce(&minDeltaT, &recDeltaT, 1, MPI_DOUBLE, MPI_MIN, velocity->comm());

    return 0.1*recDeltaT;
}

void ReactionDiffusionGrayScott(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["morphogens"];
    int nDim = subFill.nDim;                         // dimension of the space (it is always 3)
    int pDim = subFill.pDim;                         // dimension of the parametrizied object

    int numDOFs = subFill.numDOFs;
    int eNN  = subFill.eNN;

    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    wrapper<double,2> nborDOFs(subFill.nborDOFs.data(), eNN, numDOFs);
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, numDOFs);

    double jac{};
    tensor<double, 2> dbfdx(eNN, pDim);
    GlobalBasisFunctions::gradients(dbfdx, jac, subFill);

    using ttl::index::I, ttl::index::J, ttl::index::N;
    using ttl::index::a, ttl::index::b, ttl::index::c, ttl::index::i;

    tensor<double,1> c0 = nborDOFs0(N,a) * bf(N);
    tensor<double,1> c1 = nborDOFs(N,a) * bf(N);
    tensor<double,2> dcdx = dbfdx(N,a) * nborDOFs(N,i);

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, numDOFs, eNN, numDOFs);
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, numDOFs);

    const double dt  = fillStr.getRealParameter(Params::dt);
    const double dα  = fillStr.getRealParameter(Params::dA);
    const double dβ  = fillStr.getRealParameter(Params::dB);
    const double f = fillStr.getRealParameter(Params::f);
    const double k = fillStr.getRealParameter(Params::k);

    const double α0 = c0(0);
    const double α1 = c1(0);
    const double β0 = c0(1);
    const double β1 = c1(1);

    Bk(I, 0) = jac*(bf(I) * (α1-α0) / dt + dα * dcdx(a, 0) * dbfdx(I, a) + bf(I) * α1 * β1 * β1 - bf(I) * f * (1.0 - α1));
    Bk(I, 1) = jac*(bf(I) * (β1-β0) / dt + dβ * dcdx(a, 1) * dbfdx(I, a) - bf(I) * α1 * β1 * β1 + bf(I) * (f + k) * β1);

    Ak(I, 0, J, 0) = jac * (bf(I) * bf(J) / dt + dα * dbfdx(I, a) * dbfdx(J, a) + bf(I) * bf(J) * β1 * β1 + f * bf(I) * bf(J) );
    Ak(I, 0, J, 1) = jac * 2.0 * α1 * β1 * bf(I) * bf(J);
    Ak(I, 1, J, 1) = jac * (bf(I) * bf(J) / dt + dβ * dbfdx(I, a) * dbfdx(J, a) - 2.0 * α1 * β1 * bf(I) * bf(J) + (f + k) * bf(I) * bf(J) );
    Ak(I, 1, J, 0) = jac * (- β1*β1 * bf(I) * bf(J));
}

void ALEBulk(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["displacement"];
    int pDim = subFill.pDim;                         // dimension of the geometry
    int DOF = subFill.numDOFs;                       // degrees of freedom
    int eNN  = subFill.eNN;                          // number of nodes of the neighborhood

    using ttl::index::I, ttl::index::J;
    using ttl::index::a, ttl::index::b, ttl::index::c;

    wrapper<double,2> nborAux(subFill.nborAuxF.data(), eNN, subFill.numAuxF);
    tensor<double,2> nborX0 = nborAux(ttl::all, ttl::range(0,1));

    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    wrapper<double,2> dbfdξ(subFill.nborBFsGrads(), eNN, pDim);

    tensor<double,2> dxdξ  = dbfdξ(I,a) * nborX0(I,b);
    tensor<double,2> metric = dxdξ(a,c) * dxdξ(b,c);
    double jac = sqrt(metric.det());

    tensor<double,2> dξdx = dxdξ(ttl::all, ttl::range(0,1)).inv();
    tensor<double,2> dbfdx(eNN, pDim);
    dbfdx = dbfdξ(I, a) * dξdx(b, a);

    const double lame1 = fillStr.getRealParameter(Params::lame1);
    const double lame2 = fillStr.getRealParameter(Params::lame2);

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, DOF, eNN, DOF);

    Ak(I,a,J,b)  = jac * lame1 * ((dbfdx(I, c) * dbfdx(J, c)) * Identity2(a, b) + (dbfdx(I, b) * dbfdx(J, a)) );
    Ak(I,a,J,b) += jac * lame2 *  dbfdx(I, a) * dbfdx(J, b);
}

void ALEBoundary(hiperlife::FillStructure &fillStr)
{
    using ttl::tensor;
    using ttl::wrapper;
    using ttl::Identity2;
    using std::vector;
    using namespace hiperlife;

    SubFillStructure& subFill = fillStr["displacement"];
    int nDim = subFill.nDim;                         // dimension of the space (always 3)
    int pDim = subFill.pDim;                         // dimension of the geometry
    int DOF = subFill.numDOFs;                       // degrees of freedom
    int eNN  = subFill.eNN;                          // number of nodes of the neighborhood

    wrapper<double,2> nborAux(subFill.nborAuxF.data(), eNN, subFill.numAuxF);
    tensor<double,2> nborX0 = nborAux(ttl::all, ttl::range(0,1));
    wrapper<double,2> nborDOFs0(subFill.nborDOFs0.data(), eNN, DOF);

    wrapper<double,1> bf(subFill.nborBFs(), eNN);
    wrapper<double,2> dbfdξ(subFill.nborBFsGrads(), eNN, pDim);

    using ttl::index::I, ttl::index::J;
    using ttl::index::a, ttl::index::b, ttl::index::c;

    tensor<double,2> dxdξ  = dbfdξ(I,a) * nborX0(I,b);
    tensor<double,2> metric = dxdξ(a,c) * dxdξ(b,c);

    tensor<double,2> dbfdx(eNN, pDim);
    tensor<double,2> dξdx = dxdξ(ttl::all, ttl::range(0,1)).inv();
    dbfdx = dbfdξ(I, a) * dξdx(b, a);

    const double lame1 = fillStr.getRealParameter(Params::lame1);
    const double lame2 = fillStr.getRealParameter(Params::lame2);
    const double kp = fillStr.getRealParameter(Params::kp);


    vector<double> elem_tangent_vec = subFill.tangentsBoundaryRef();          // in element coordinates
    vector<double> elem_normal_vec = subFill.normalsBoundaryRef();            // in element coordinates
    wrapper<double,1> elem_tangent(elem_tangent_vec.data(), pDim);  // in element coordinates
    wrapper<double,1> elem_normal(elem_normal_vec.data(), pDim);    // in element coordinates

    wrapper<double,2> nborCoords(subFill.nborCoords.data(), eNN, nDim);
    tensor<double,2> dx0dξ = nborX0(I, b) * dbfdξ(I, a);                      // d(x0) / dξ
    tensor<double,2> dxNdξ = (nborX0(I, b) + nborDOFs0(I, b)) * dbfdξ(I, a);  // d(x0+u) / dξ

    tensor<double,1> tangent0 = elem_tangent(a) * dx0dξ(b, a);
    const double dl = tangent0.norm();
    tangent0 /= dl;
    tensor<double,1> normal0 = {tangent0(1),-tangent0(0)};
    // normal0 = normal0/normal0.norm();

    tensor<double,1> tangentN = elem_tangent(a) * dxNdξ(b, a);
    tangentN /= tangentN.norm();
    tensor<double,1> normalN = {tangentN(1),-tangentN(0)};
    // normalN = normalN/normalN.norm();

    wrapper<double,4> Ak(fillStr.Ak(0, 0).data(), eNN, DOF, eNN, DOF);
    wrapper<double,2> Bk(fillStr.Bk(0).data(), eNN, DOF);

    tensor<double,2> Aux = lame1 * dbfdx(J, c) * normalN(c) * normal0(a) +
            + lame1 * dbfdx(J, c) * normalN(a) * normal0(c)
            + lame2 * dbfdx(J, a) * normalN(c) * normal0(c);

    Ak(I,a,J,b) -= dl * bf(I) * normalN(a) * Aux(J,b);
    Ak(I,a,J,b) -= dl * Aux(I,a) * bf(J) * normalN(b);
    Ak(I,a,J,b) += dl * kp * bf(I) * normalN(a) * bf(J) * normalN(b);
    // Ak(I,a,J,b) += dl * kp * bf(I) * bf(J) * Identity2(a,b);

    tensor<double,1> uN = bf(I) * nborDOFs0(I, a);
    const double dt = fillStr.getRealParameter(Params::dt);

    tensor<double,1> vel = bf(I) * nborAux(ttl::all, ttl::range(2,3))(I, a);

    Bk(I, a) -= dl * (uN(b) + vel(b) * dt) * normalN(b) * Aux(I,a);
    Bk(I, a) += dl * kp *  (uN(b) + vel(b) * dt) * normalN(b) * bf(I) * normalN(a);

    Bk(I, a) += dl * 1.E-4 *  (uN(b) + vel(b) * dt) * tangentN(b) * bf(I) * tangentN(a);
    Ak(I,a,J,b) += dl * 1.E-4 * bf(I) * tangentN(a) * bf(J) * tangentN(b);

}

void DeformMesh(const hiperlife::SmartPtr<hiperlife::LinearSolver>& linSolver, const hiperlife::SmartPtr<hiperlife::DOFsHandler>& deformation)
{
    using namespace hiperlife;

    deformation->nodeDOFs0->setValue(deformation->nodeDOFs);
    deformation->UpdateGhosts();
    linSolver->solve();
    linSolver->UpdateSolution();

    if(!linSolver->converged()) {
        std::cerr << "DeformMesh failed. Solver has not converged!" << std::endl;
        abort();
    }

    const double dt = linSolver->hiperProblem()->userStructure()->dparam[Params::dt];
    for(int i = 0; i < deformation->mesh->loc_nPts(); i++) {
        const double xN = deformation->mesh->_nodeData->getValue(0, i, IndexType::Local);
        const double uxN = deformation->nodeDOFs0->getValue(0, i, IndexType::Local);
        const double ux = deformation->nodeDOFs->getValue(0, i, IndexType::Local);
        const double xN1 = xN + ux - uxN;
        deformation->mesh->_nodeData->setValue(0, i, IndexType::Local, xN1);

        const double yN = deformation->mesh->_nodeData->getValue(1, i, IndexType::Local);
        const double uyN = deformation->nodeDOFs0->getValue(1, i, IndexType::Local);
        const double uy = deformation->nodeDOFs->getValue(1, i, IndexType::Local);
        const double yN1 = yN + uy - uyN;
        deformation->mesh->_nodeData->setValue(1, i, IndexType::Local, yN1);

        const double vx = deformation->nodeAuxF->getValue("vx", i, IndexType::Local);
        const double vy = deformation->nodeAuxF->getValue("vy", i, IndexType::Local);
        double norm = sqrt((ux - uxN)*(ux - uxN) + (uy - uyN) * (uy - uyN));
        const double errUx = ux - uxN - vx * dt;
        const double errUy = uy - uyN - vy * dt;
        deformation->nodeAuxF->setValue("errUx", i, IndexType::Local, errUx/norm);
        deformation->nodeAuxF->setValue("errUy", i, IndexType::Local, errUy/norm);

    }
    deformation->mesh->_nodeData->UpdateGhosts();
    deformation->UpdateGhosts();
}
