#pragma once

#include <hl_LinearSolver.h>

struct Params
{
    enum RealParameters
    {
        dt,
        dA,
        dB,
        k,
        dc,
        dh,
        rhoc,
        rhoh,
        c0,
        cs,
        m0,
        gamma,
        mu,
        nu,
        lame1,
        lame2,
        kp,
        f
    };

    enum StringParameters
    {
        filemesh,
        mumpsanalysis,
        consistency,
        prefix
    };

    HL_PARAMETER_LIST DefaultValues{
            {"dt",1.E-3},
            {"dc",0.005},
            {"dh",0.05},
            {"rhoc",1.0},
            {"rhoh",1.5},
            {"k",1.0},
            {"cs",1.0},
            {"c0",1.0},
            {"m0",1.0},
            {"gamma",5.0},
            {"mu",1.0},
            {"nu",1.E-2},
            {"lame1",1.0},
            {"lame2",0.1},
            {"kp",1.E3},
            {"filemesh",""},
            {"prefix", "field"},
            {"mumpsanalysis","parallel", {"sequential","parallel"}},
            {"consistency","none",{"none","hessian"}}
    };
};

void ReactionDiffusion(hiperlife::FillStructure &fillStr);

void ConvectionDiffusion(hiperlife::FillStructure &fillStr);

void ConvectionDiffusionALE(hiperlife::FillStructure &fillStr);

void TensionFlow(hiperlife::FillStructure &fillStr);

double CheckCFL(hiperlife::SmartPtr<hiperlife::DOFsHandler>& velocity);

void ReactionDiffusionGrayScott(hiperlife::FillStructure &fillStr);

void ALEBulk(hiperlife::FillStructure &fillStrr);

void ALEBoundary(hiperlife::FillStructure &fillStr);

void DeformMesh(const Teuchos::RCP<hiperlife::LinearSolver>& linSolver, const hiperlife::SmartPtr<hiperlife::DOFsHandler>& deformation);


