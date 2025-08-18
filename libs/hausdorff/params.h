#pragma once
#include "3rdParty/optix7support.h"

enum OptiXHDistOptimizeState {
    FILTERING,
    COMPUTING
};

struct OptiXHDistParam{
    float3* query;
    size_t querySize;

    float3* target;
    float3* representative;
    unsigned int* targetIDX;
    float* distance;

    OptixAabb targetBound;
    float3 targetVoxelSize;
    size_t* clusterInfo;

    OptiXHDistOptimizeState state = COMPUTING;

    float targetEPS = 0.5;
    float3 targetTransform = { 0.0f, 0.0f, 0.0f };

    OptixTraversableHandle traversable;
};