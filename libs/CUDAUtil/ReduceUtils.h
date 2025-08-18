#pragma once
#include "3rdParty/helper_math.h"
#include "3rdParty/CUDABuffer.h"

float getMaximumF(float* value, size_t size, size_t& idx);
float getMinimumF(float* value, size_t size, size_t& idx);

/// <summary>
/// 
/// </summary>
/// <param name="source">Source points</param>
/// <param name="source_size">size of source</param>
/// <param name="distance">reference distance</param>
/// <param name="result">reduced points</param>
/// <param name="new_size">size of new points</param>
/// <param name="op">binary operator true or false (check is hit with distance > 0)</param>
void reduceWithDistance(float3* source, size_t source_size, float* distance, float3*& result, size_t& new_size, const bool op);
void reduceWithDistance_2(float3* source, uint* idxs, size_t source_size, float* distance, float3*& result, uint*& idxsResult, size_t& new_size, const bool op);

void countTrianglesPerVertex(float3* vtx, size_t vtx_size, uint3* tris, size_t tri_size, uint* res);
void makeVertexTrianlgeMap(float3* vtx, size_t vtx_size, uint3* tris, size_t tri_size, uint* res, uint* map, uint& map_size, uint* ref);
void getTriangleList(uint3* origin, int* ptrInfo, uint3* resTri, size_t vSize, size_t tSize, size_t& new_size);
void cudaSubdivide(float3* originVTX, uint3* originTRI, uint3* subTRI, size_t subTRISize, float3*& outputVTX, uint3*& outputTRI, size_t& outputVSize, size_t& outputTSize);