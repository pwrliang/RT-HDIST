#pragma once
#include "3rdParty/helper_math.h"
#include "3rdParty/CUDABuffer.h"

OptixAabb computeAABB_device(float3* vert, size_t vertSize);
void genUniformClusterAabb(float3* d_points, size_t pointSize, OptixAabb*& cluster, float3*& voxelIDs, size_t& number_of_cluster, std::vector<size_t>& vertex_pos_per_cluster, BYTE bitCount);
void genUniformClusterAabb(float3* d_points, size_t pointSize, uint3* tris, OptixAabb*& cluster, float3*& voxelIDs, size_t& number_of_cluster, std::vector<size_t>& vertex_pos_per_cluster, BYTE bitCount);
void genAABBs(float3* points, float radius, size_t numPoints, OptixAabb* d_aabb);

void genCenterPointsOfTris(float3* gpuBuff, size_t numTris, float3* vtxBuff, std::vector<uint3>& tris);
void genCenterPointsOfTris(float3*& gpuBuff, size_t numTris, float3* vtxBuff, uint3* trisGPU);
void reduceAndGetLUT(float3* vtx, size_t numVtx, float3** onIndexSpace, uint** LUT, uint** idxIDXquery, size_t& uniqueSize, OptixAabb bounds, float3 voxelSize);

void ToIndexSpace(float3* vtx, size_t numVtx, float3* onIndexSpace, OptixAabb bounds, float3 voxelSize);
void voxelize_cpu(float3* vtx, size_t numVtx, uint3* idx, size_t numTris, float3*& voxelIDs, size_t& num_of_cluster, uint*& idxList, std::vector<size_t>& index_pos_per_cluster, BYTE bitCount);

void overlapCulling(float3* vertA, size_t sizeA, float3* vertB, size_t sizeB, float3*& res, size_t& sizeRes, BYTE bitCount);