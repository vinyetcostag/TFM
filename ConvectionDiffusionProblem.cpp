#include <hl_LinearSolver_Iterative_Belos.h>
#include <hl_NonlinearSolver_NewtonRaphson.h>
#include <hl_LinearSolver_Direct_MUMPS.h>
#include <random>
#include "hl_DistributedClass.h"
#include "hl_HiPerProblem.h"
#include "hl_StructMeshGenerator.h"
#include "hl_ConsistencyCheck.h"
#include "hl_ParamStructure.h"
#include "hl_Parser.h"

#include "Physics.h"

int main(int argc, char** argv) {
    using std::cout, std::cerr;
    using Teuchos::RCP, Teuchos::rcp;
    using namespace hiperlife;

    hiperlife::Init(argc, argv);

    SmartPtr<ParamStructure> paramStr = ReadParamsFromCommandLine<Params>();

    SaveParamsToConfigFile(paramStr, paramStr->getStringParameter(Params::prefix) + "_config.txt");

    SmartPtr<StructMeshGenerator> meshGen = Create<StructMeshGenerator>();
    meshGen->setMesh(ElemType::Triang, BasisFuncType::Lagrangian, 1);
    meshGen->setPeriodicBoundaryCondition({Axis::Xaxis, Axis::Yaxis});
    // meshGen->genSquare(50, 2.0);
    meshGen->genRectangle(1000, 1, 2.0, 2.0);

    SmartPtr<DistributedMesh> mesh = Create<DistributedMesh>();
    mesh->setMesh(meshGen);
    mesh->setBalanceMesh(true);
    mesh->Update();

    SmartPtr<DOFsHandler> fieldMorphogens= Create<DOFsHandler>(mesh);
    fieldMorphogens->setNameTag("morphogens");
    fieldMorphogens->setDOFs({"c", "h", "m"});
    fieldMorphogens->setNodeAuxF({"vx", "vy"});
    fieldMorphogens->Update();

    SmartPtr<HiPerProblem> problem = Create<HiPerProblem>();
    problem->setParameterStructure(paramStr);
    problem->setDOFsHandlers({fieldMorphogens});
    problem->setIntegration("IntegMorphogens", {"morphogens"});
    problem->setCubatureGauss("IntegMorphogens", 3);
    if (paramStr->getStringParameter(Params::consistency) == "none") {
        problem->setElementFillings("IntegMorphogens", ConvectionDiffusion);
    }else if (paramStr->getStringParameter(Params::consistency) == "full") {
        problem->setElementFillings("IntegMorphogens", ConsistencyCheck<ConvectionDiffusion>);
        problem->setConsistencyCheckDelta(1.E-4);
        problem->setConsistencyCheckTolerance(1.E-4);
        problem->setConsistencyCheckType(ConsistencyCheckType::Both);
    }    else if (paramStr->getStringParameter(Params::consistency) == "jacobian") {
        problem->setElementFillings("IntegMorphogens", ConsistencyCheck<ConvectionDiffusion>);
        problem->setConsistencyCheckDelta(1.E-4);
        problem->setConsistencyCheckTolerance(1.E-4);
        problem->setConsistencyCheckType(ConsistencyCheckType::Hessian);
    }
        problem->Update();

    if(problem->myRank()==0){cout << "MUMPS analysis type: " << paramStr->getStringParameter(Params::mumpsanalysis) << endl;}

    std::random_device rd;  // seed the random number generator
    std::mt19937 gen(rd()); // use the Mersenne Twister 19937 generator
    std::uniform_real_distribution<double> dis(-0.01, 0.01); // create a uniform distribution of doubles between 0 and 1
    fieldMorphogens->setInitialCondition("c", [&gen, &dis](double x, double y){
        return 1.0 + dis(gen);
    });
    fieldMorphogens->setInitialCondition("h", [&dis, &gen](double x, double y) {
        return 1.0 + dis(gen);
    });
    fieldMorphogens->setInitialCondition("m", 1.0);

    fieldMorphogens->printFileVtk(paramStr->getStringParameter(Params::prefix) + "_morphogens_0", true);

    SmartPtr<MUMPSDirectLinearSolver> linSolReactionDiff = Create<MUMPSDirectLinearSolver>();
    linSolReactionDiff->setHiPerProblem(problem);
    linSolReactionDiff->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    linSolReactionDiff->setDefaultParameters();
    if(paramStr->getStringParameter(Params::mumpsanalysis) == "sequential") {
        linSolReactionDiff->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Sequential);
    }else{
        linSolReactionDiff->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel);
    }
    linSolReactionDiff->Update();

    SmartPtr<NewtonRaphsonNonlinearSolver> nonLinSolReactionDiff = Create<NewtonRaphsonNonlinearSolver>();
    nonLinSolReactionDiff->setLinearSolver(linSolReactionDiff);
    nonLinSolReactionDiff->setMaxNumIterations(5);
    nonLinSolReactionDiff->setResTolerance(1.E-8);
    nonLinSolReactionDiff->setSolTolerance(1.E-8);
    nonLinSolReactionDiff->setLineSearch(false);
    nonLinSolReactionDiff->setConvRelTolerance(false);
    nonLinSolReactionDiff->setPrintIntermInfo(true);
    nonLinSolReactionDiff->setPrintSummary(false);
    nonLinSolReactionDiff->Update();
    nonLinSolReactionDiff->solutionError();

    SmartPtr<DOFsHandler> fieldVelocity = Create<DOFsHandler>(mesh);
    fieldVelocity->setNameTag("velocity");
    fieldVelocity->setDOFs({"vx", "vy"});
    fieldVelocity->setNodeAuxF({"c", "h", "m"});
    fieldVelocity->Update();

    fieldMorphogens->nodeAuxF->mirrorField(0, 0, fieldVelocity->nodeDOFs);
    fieldMorphogens->nodeAuxF->mirrorField(1, 1, fieldVelocity->nodeDOFs);
    fieldVelocity->nodeAuxF->mirrorField(0, 0, fieldMorphogens->nodeDOFs);
    fieldVelocity->nodeAuxF->mirrorField(1, 1, fieldMorphogens->nodeDOFs);
    fieldVelocity->nodeAuxF->mirrorField(2, 2, fieldMorphogens->nodeDOFs);


    SmartPtr<HiPerProblem> problemFlow = Create<HiPerProblem>();
    problemFlow->setParameterStructure(paramStr);
    problemFlow->setDOFsHandlers({fieldVelocity});
    problemFlow->setIntegration("IntegFlow", {"velocity"});
    problemFlow->setCubatureGauss("IntegFlow", 3);
    problemFlow->setElementFillings("IntegFlow", TensionFlow);
    problemFlow->Update();

    SmartPtr<MUMPSDirectLinearSolver> linSolFlow = Create<MUMPSDirectLinearSolver>();
    linSolFlow->setHiPerProblem(problemFlow);
    linSolFlow->setVerbosity(MUMPSDirectLinearSolver::Verbosity::Medium);
    linSolFlow->setDefaultParameters();
    if(paramStr->getStringParameter(Params::mumpsanalysis) == "sequential") {
        linSolFlow->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Sequential);
    }else{
        linSolFlow->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel);
    }
    linSolFlow->Update();

    fieldVelocity->nodeAuxF->setValue(fieldMorphogens->nodeDOFs);
    problemFlow->UpdateGhosts();
    linSolFlow->solve();
    linSolFlow->UpdateSolution();
    fieldVelocity->printFileVtk(paramStr->getStringParameter(Params::prefix) +"_velocity_0", true);

    double &dt = paramStr->getRealParameter(Params::dt);

    for(int i = 1; i < 8000; i++) {
        if(problem->myRank() == 0)
            std::cout << "step: " << i << " : dt: " << dt << endl;

        fieldMorphogens->nodeDOFs0->setValue(fieldMorphogens->nodeDOFs);
        fieldMorphogens->UpdateGhosts();

        nonLinSolReactionDiff->solve();

        if(nonLinSolReactionDiff->converged()){
            if(nonLinSolReactionDiff->numberOfIterations() <= 5)
                dt *= 1.1;
            else if(nonLinSolReactionDiff->numberOfIterations() > 7)
                dt *= 0.9;

            std::string fileI = paramStr->getStringParameter(Params::prefix) + "_morphogens_" + to_string(i);
            fieldMorphogens->printFileVtk(fileI, true);
        }
        else {
            fieldMorphogens->nodeDOFs->setValue(fieldMorphogens->nodeDOFs0);
            dt *= 0.8;
            i--;

            if(dt < 1e-10)
                break;

            continue;
        }

        problemFlow->UpdateGhosts();

        linSolFlow->solve();
        linSolFlow->UpdateSolution();

        const double cflDt = CheckCFL(fieldVelocity);
        if(cflDt < dt) {
            const double newDt = 0.9*cflDt;
            if(problem->myRank() == 0)
                std::cout << "Warning!!! cfl condition is not satisfied, decreasing time step. Current dt: " << i << " : dt: " << paramStr->getRealParameter(Params::dt) << " -> new dt: " << newDt << endl;
            dt = newDt;
        }

        std::string fileI = paramStr->getStringParameter(Params::prefix) + "_velocity_" + to_string(i);
        fieldVelocity->printFileVtk(fileI, true); 
    }

    hiperlife::Finalize();
}
