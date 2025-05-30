#include <random>

#include <hl_LinearSolver_Iterative_Belos.h>
#include <hl_NonlinearSolver_NewtonRaphson.h>
#include <hl_LinearSolver_Direct_MUMPS.h>
#include <hl_UnstructVtkMeshGenerator.h>
#include <hl_MeshLoader.h>
#include "hl_DistributedClass.h"
#include "hl_HiPerProblem.h"
#include "hl_ConsistencyCheck.h"
#include "hl_ParamStructure.h"
#include "hl_Parser.h"

#include "Physics.h"

int main(int argc, char** argv) {
    using std::cout, std::cerr;
    using namespace hiperlife;

    hiperlife::Init(argc, argv);

    SmartPtr<ParamStructure> paramStr = ReadParamsFromCommandLine<Params>();

    SaveParamsToConfigFile(paramStr, paramStr->getStringParameter(Params::prefix) + "_config.txt");

    SmartPtr<MeshCreator> meshCreator;
    if(paramStr->getStringParameter(Params::filemesh) == "") {
        SmartPtr<UnstructVtkMeshGenerator> meshGen = Create<UnstructVtkMeshGenerator>();
        meshGen->setMesh(hiperlife::ElemType::Triang, BasisFuncType::Linear, 1);
        const double h = 0.002;
        meshGen->genPolygon([h](double x[3]){ return h;});
        meshCreator = meshGen;
    }
    else {
        std::string meshFile(paramStr->getStringParameter(Params::filemesh));
        SmartPtr<MeshLoader> meshLoader = Create<MeshLoader>();
        meshLoader->setMesh(hiperlife::ElemType::Triang, BasisFuncType::Linear, 1);
        meshLoader->loadVtk(meshFile, MeshType::Parallel);
        meshCreator = meshLoader;
    }


    SmartPtr<DistributedMesh> mesh = Create<DistributedMesh>();
    mesh->setMesh(meshCreator);
    mesh->setBalanceMesh(true);
    mesh->Update();

    SmartPtr<DOFsHandler> fieldMorphogens = Create<DOFsHandler>(mesh);
    fieldMorphogens->setNameTag("morphogens");
    fieldMorphogens->setDOFs({"c", "h", "m"});
    fieldMorphogens->setNodeAuxF({"vx", "vy", "ux", "uy", "uxN", "uyN"});
    fieldMorphogens->Update();

    SmartPtr<DOFsHandler> fieldVelocity = Create<DOFsHandler>(mesh);
    fieldVelocity->setNameTag("velocity");
    fieldVelocity->setDOFs({"vx", "vy"});
    fieldVelocity->setNodeAuxF({"c", "h", "m"});
    fieldVelocity->Update();

    SmartPtr<DOFsHandler> fieldDisplacement = Create<DOFsHandler>(mesh);
    fieldDisplacement->setNameTag("displacement");
    fieldDisplacement->setDOFs({"ux", "uy"});
    fieldDisplacement->setNodeAuxF({"x0", "y0", "vx", "vy", "errUx", "errUy"});
    fieldDisplacement->Update();
    fieldDisplacement->setInitialCondition(0, 0.0);
    fieldDisplacement->setInitialCondition(1, 0.0);
    fieldDisplacement->UpdateGhosts();

    fieldMorphogens->nodeAuxF->mirrorField(0, 0, fieldVelocity->nodeDOFs);
    fieldMorphogens->nodeAuxF->mirrorField(1, 1, fieldVelocity->nodeDOFs);
    fieldMorphogens->nodeAuxF->mirrorField(2, 0, fieldDisplacement->nodeDOFs);
    fieldMorphogens->nodeAuxF->mirrorField(3, 1, fieldDisplacement->nodeDOFs);
    fieldMorphogens->nodeAuxF->mirrorField(4, 0, fieldDisplacement->nodeDOFs0);
    fieldMorphogens->nodeAuxF->mirrorField(5, 1, fieldDisplacement->nodeDOFs0);

    fieldVelocity->nodeAuxF->mirrorField(0, 0, fieldMorphogens->nodeDOFs);
    fieldVelocity->nodeAuxF->mirrorField(1, 1, fieldMorphogens->nodeDOFs);
    fieldVelocity->nodeAuxF->mirrorField(2, 2, fieldMorphogens->nodeDOFs);

    fieldDisplacement->nodeAuxF->setValue(0, 0, mesh->_nodeData);
    fieldDisplacement->nodeAuxF->setValue(1, 1, mesh->_nodeData);
    fieldDisplacement->nodeAuxF->mirrorField(2, 0, fieldVelocity->nodeDOFs);
    fieldDisplacement->nodeAuxF->mirrorField(3, 1, fieldVelocity->nodeDOFs);
    fieldDisplacement->UpdateGhosts();

    SmartPtr<HiPerProblem> problem = Create<HiPerProblem>();
    problem->setParameterStructure(paramStr);
    problem->setDOFsHandlers({fieldMorphogens});
    problem->setIntegration("IntegMorphogens", {"morphogens"});
    problem->setCubatureGauss("IntegMorphogens", 3);
    if (paramStr->getStringParameter(Params::consistency) == "none") {
        problem->setElementFillings("IntegMorphogens", ConvectionDiffusionALE);
    }    else if (paramStr->getStringParameter(Params::consistency) == "hessian") {
        problem->setElementFillings("IntegMorphogens", ConsistencyCheck<ConvectionDiffusionALE>);
        problem->setConsistencyCheckDelta(1.E-4);
        problem->setConsistencyCheckTolerance(1.E-4);
        problem->setConsistencyCheckType(ConsistencyCheckType::Hessian);
    }
    problem->setGlobalIntegrals({"area","mass"});
    problem->Update();
    problem->FillLinearSystem();
    problem->globalIntegral("mass"); // devuelve mass en step 0
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

    fieldMorphogens->printFileVtk("fieldMorphogens0", true);

    SmartPtr<MUMPSDirectLinearSolver> linSolReactionDiff = Create<MUMPSDirectLinearSolver>();
    linSolReactionDiff->setHiPerProblem(problem);
    linSolReactionDiff->setVerbosity(MUMPSDirectLinearSolver::Verbosity::None);
    linSolReactionDiff->setDefaultParameters();
    if(paramStr->getStringParameter(Params::mumpsanalysis) == "sequential") {
        linSolReactionDiff->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Sequential);
    }else
        linSolReactionDiff->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel);
    linSolReactionDiff->Update();

    SmartPtr<NewtonRaphsonNonlinearSolver> nonLinSolReactionDiff = Create<NewtonRaphsonNonlinearSolver>();
    nonLinSolReactionDiff->setLinearSolver(linSolReactionDiff);
    nonLinSolReactionDiff->setMaxNumIterations(100);
    nonLinSolReactionDiff->setResTolerance(1.E-8);
    nonLinSolReactionDiff->setSolTolerance(1.E-8);
    nonLinSolReactionDiff->setLineSearch(false);
    nonLinSolReactionDiff->setConvRelTolerance(true);
    nonLinSolReactionDiff->setPrintIntermInfo(true);
    nonLinSolReactionDiff->setPrintSummary(true);
    nonLinSolReactionDiff->Update();

    SmartPtr<HiPerProblem> problemFlow = Create<HiPerProblem>();
    problemFlow->setParameterStructure(paramStr);
    problemFlow->setDOFsHandlers({fieldVelocity});
    problemFlow->setIntegration("IntegFlow", {"velocity"});
    problemFlow->setCubatureGauss("IntegFlow", 3);
    problemFlow->setElementFillings("IntegFlow", TensionFlow);
    problemFlow->Update();

    SmartPtr<MUMPSDirectLinearSolver> linSolFlow = Create<MUMPSDirectLinearSolver>();
    linSolFlow->setHiPerProblem(problemFlow);
    linSolFlow->setVerbosity(MUMPSDirectLinearSolver::Verbosity::Extreme);
    linSolFlow->setDefaultParameters();
    if(paramStr->getStringParameter(Params::mumpsanalysis) == "sequential") {
        linSolFlow->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Sequential);
    }else{
        linSolFlow->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel);

    }
    linSolFlow->Update();

    problemFlow->UpdateGhosts();
    linSolFlow->solve();
    linSolFlow->UpdateSolution();
    fieldVelocity->printFileVtk(paramStr->getStringParameter(Params::prefix) + "_velocity_0", true);


    SmartPtr<HiPerProblem> problemDispl = Create<HiPerProblem>();
    problemDispl->setParameterStructure(paramStr);
    problemDispl->setDOFsHandlers({fieldDisplacement});
    problemDispl->setIntegration("IntegALEBulk", {"displacement"});
    problemDispl->setCubatureGauss("IntegALEBulk", 3);
    problemDispl->setElementFillings("IntegALEBulk", ALEBulk);
    problemDispl->setIntegration("IntegALEBoundary", {"displacement"});
    problemDispl->setCubatureBorderGauss("IntegALEBoundary", 3);
    problemDispl->setElementFillings("IntegALEBoundary", ALEBoundary);
    problemDispl->Update();

    SmartPtr<MUMPSDirectLinearSolver> linSolDispl = Create<MUMPSDirectLinearSolver>();
    linSolDispl->setHiPerProblem(problemDispl);
    linSolDispl->setVerbosity(MUMPSDirectLinearSolver::Verbosity::Extreme);
    linSolDispl->setDefaultParameters();
    if(paramStr->getStringParameter(Params::mumpsanalysis) == "sequential") {
        linSolDispl->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Sequential);
    }else{
        linSolDispl->setAnalysisType(MUMPSDirectLinearSolver::AnalysisType::Parallel);
    }
    linSolDispl->Update();

    // DeformMesh(linSolDispl, fieldDisplacement);

    fieldDisplacement->printFileVtk(paramStr->getStringParameter(Params::prefix) + "displacement_0", true);

    double& dt = paramStr->getRealParameter(Params::dt);
    for(int i = 1; i < 1000; i++) {
        if(problem->myRank() == 0)
            std::cout << "step: " << i << " : dt: " << dt << endl;

        fieldMorphogens->nodeDOFs0->setValue(fieldMorphogens->nodeDOFs);
        nonLinSolReactionDiff->solve();
        // pintar area y masa. 
        double mass = problem->globalIntegral("mass");
        if (problem->myRank() == 0) {
            std::ofstream out("mass_history.txt", std::ios::app);
            out << i << "\t" << mass << "\n";
        }
        if(nonLinSolReactionDiff->converged()){
            if(nonLinSolReactionDiff->numberOfIterations() <= 4)
                dt *= 1.1;
            else if(nonLinSolReactionDiff->numberOfIterations() > 7)
                dt *= 0.9;
        }
        else {
            fieldMorphogens->nodeDOFs->setValue(fieldMorphogens->nodeDOFs0);
            dt *= 0.8;
            i--;

            if(dt < 1e-10)
                break;

            continue;
        }

        std::string fileCI = paramStr->getStringParameter(Params::prefix) + "_morphogens_" + to_string(i);
        fieldMorphogens->printFileVtk(fileCI, true);

        linSolFlow->solve();
        linSolFlow->UpdateSolution();

        if(problem->myRank() == 0)
            cout << "I solved for velocities!!!!" << endl;

        std::string fileVI = "fieldVelocity" + to_string(i);
        fieldVelocity->printFileVtk(fileVI, true);

        DeformMesh(linSolDispl, fieldDisplacement);
        std::string fileDI = paramStr->getStringParameter(Params::prefix) + "_displacement_" + to_string(i);
        fieldDisplacement->printFileVtk(fileDI, true);

        const double cflDt = CheckCFL(fieldVelocity);
        if(cflDt < dt) {
            const double newDt = 0.9*cflDt;
            if(problem->myRank() == 0)
                std::cout << "Warning!!! cfl condition is not satisfied, decreasing time step. Current dt: " << i << " : dt: " << paramStr->getRealParameter(Params::dt) << " -> new dt: " << newDt << endl;
            paramStr->setRealParameter(Params::dt, newDt);
        }
    }

    hiperlife::Finalize();
}
