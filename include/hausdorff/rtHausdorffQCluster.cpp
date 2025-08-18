#include "rtHausdorffQCluster.h"
#include "params.h"
#include "MortonUtils.h"
#include "ReduceUtils.h"

#include "3rdParty/TimeChecker.h"

#include "AABBSupport.h"

#include <iostream>

/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) DummyRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void* data;
};

OptiXHDProgram::OptiXHDProgram(OptiXProgramCompileOption programOption) : OptiXPrograms(programOption) {
    //Set SBT because HD not needed SBT
    
    // ------------------------------------------------------------------
    // build raygen records
    // ------------------------------------------------------------------
    std::vector<DummyRecord> raygenRecords;
    for (int i = 0; i < raygenPGs.size(); i++) {
        DummyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        raygenRecords.push_back(rec);
    }
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

    // ------------------------------------------------------------------
    // build miss records
    // ------------------------------------------------------------------
    std::vector<DummyRecord> missRecords;
    for (int i = 0; i < missPGs.size(); i++) {
        DummyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }
    missRecordsBuffer.alloc_and_upload(missRecords);
    sbt.missRecordBase = missRecordsBuffer.d_pointer();
    sbt.missRecordStrideInBytes = sizeof(DummyRecord);
    sbt.missRecordCount = (int)missRecords.size();

    // ------------------------------------------------------------------
    // build hitgroup records
    // ------------------------------------------------------------------
    std::vector<DummyRecord> hitgroupRecords;

    // all meshes use the same code, so all same hit group
    for (int i = 0; i < hitgroupPGs.size(); i++) {
        DummyRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        hitgroupRecords.push_back(rec);
    }

    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
    sbt.hitgroupRecordStrideInBytes = sizeof(DummyRecord);
    sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
}

//----------------------------------------------------------------------------------------------------------------------------------------//
float qclusterHD(OptiXHDProgram& program, HDGPUParam<HDMODE::POINT> A, HDGPUParam<HDMODE::POINT> B, float3& cand1, float3& cand2, float _eps, BYTE bitCount, std::map<std::string, float>& timeParam) {
    AABBCluster target;
    OptixAabb targetAABB = computeAABB_device(B.vert, B.vSize);
    OptixAabb sourceAABB = computeAABB_device(A.vert, A.vSize);

    OptixAabb totalAABB = merge(sourceAABB, targetAABB);

    const float3 voxelS = (aabb2max(targetAABB) - aabb2min(targetAABB)) / (1 << bitCount);
    float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
    const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

    target.number_of_points = B.vSize;
    cudaMalloc(&target.points, sizeof(float3) * target.number_of_points);
    cudaMemcpy(target.points, B.vert, sizeof(float3) * B.vSize, cudaMemcpyDeviceToDevice);

    cudaMalloc(&target.pointAabb, sizeof(OptixAabb) * target.number_of_points);

    float3* gpuRepresentative;
    auto TimeGenerateIndexSpace = SPIN::TimeCheck([&]() {
        genUniformClusterAabb(target.points, target.number_of_points, target.cluster, gpuRepresentative, target.number_of_cluster, target.clusterInfo, bitCount);
        });
    timeParam["IndexSpaceBuildTime"] += TimeGenerateIndexSpace;
    std::cout << "Number of cluster : " << target.number_of_cluster << std::endl;

    target.representative.resize(target.number_of_cluster);
    cudaMemcpy(target.representative.data(), gpuRepresentative, sizeof(float3) * target.number_of_cluster, cudaMemcpyDeviceToHost);

    float hDist = 0.;
    CUDABuffer launchParamBuffer;
    launchParamBuffer.alloc(sizeof(OptiXHDistParam));

    float eps = _eps;

    size_t remains = A.vSize;
    size_t previous_remains = remains;
    size_t pprevious_remains = remains;

    float3* queries;
    cudaMalloc(&queries, sizeof(float3) * remains);
    cudaMemcpy(queries, A.vert, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);

    float3* previous;
    cudaMalloc(&previous, sizeof(float3) * remains);

    float3* pprevious;
    cudaMalloc(&pprevious, sizeof(float3) * remains);

    size_t* gpuClusterInfo;
    cudaMalloc(&gpuClusterInfo, sizeof(size_t) * target.clusterInfo.size());
    cudaMemcpy(gpuClusterInfo, target.clusterInfo.data(), sizeof(size_t) * target.clusterInfo.size(), cudaMemcpyHostToDevice);

    OptixAabb* aabbBuffer;
    cudaMalloc(&aabbBuffer, sizeof(OptixAabb) * target.number_of_cluster);

    OptiXHDistParam nnparam;
    nnparam.querySize = remains;
    nnparam.representative = gpuRepresentative;
    nnparam.targetBound = targetAABB;

    ToIndexSpace(A.vert, A.vSize, queries, targetAABB, voxelSize);
    //ToIndexSpace(culledA, remains, queries, targetAABB, voxelSize);
    nnparam.query = queries;

    cudaMemcpy(previous, queries, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);
    cudaMemcpy(pprevious, queries, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);

    float ceneterLength = length(aabb2center(targetAABB) - aabb2center(sourceAABB));
    float aabblength = length(distance(sourceAABB, targetAABB));

    eps = (int)(ceneterLength / delimiter * 0.5f) * sqrtf(3);

    uint3 number_of_voxels_maximum_inspace = make_uint3(aabb2size(totalAABB) / voxelSize);
    nnparam.targetVoxelSize = voxelSize;
    nnparam.target = target.points;
    nnparam.clusterInfo = gpuClusterInfo;

    AABB_GAS targetGAS;
    targetGAS.build(aabbBuffer, target.number_of_cluster);

    while (remains > 0) {

        float* distanceBuffer;
        uint* idxBuffer;
        float3* reduced;
        size_t reduced_size;
        auto TimeFilteringStep = SPIN::TimeCheck([&]() {
            genAABBs(gpuRepresentative, eps, target.representative.size(), aabbBuffer);
            cudaDeviceSynchronize();

            targetGAS.refit(aabbBuffer, target.representative.size());

            nnparam.traversable = targetGAS.handle;
            nnparam.targetEPS = eps;

            cudaMalloc(&distanceBuffer, sizeof(float) * remains);
            cudaMalloc(&idxBuffer, sizeof(uint) * remains);

            cudaMemset(distanceBuffer, 0, sizeof(float) * remains);

            nnparam.querySize = remains;
            nnparam.distance = distanceBuffer;
            nnparam.targetIDX = idxBuffer;
            nnparam.state = FILTERING;

            launchParamBuffer.upload(&nnparam, 1);
            int sqrtSize = ceil(sqrt(remains));
            auto traverseTime = SPIN::TimeCheck([&]() {
                OPTIX_CHECK(optixLaunch(program.pipeline, 0,
                    launchParamBuffer.d_pointer(),
                    launchParamBuffer.sizeInBytes,
                    &program.sbt,
                    sqrtSize,
                    sqrtSize,
                    1
                ));
                cudaStreamSynchronize(0);
                });

            eps += sqrt(3);
            reduceWithDistance(queries, remains, nnparam.distance, reduced, reduced_size, false);
            cudaMemcpy(pprevious, previous, sizeof(float3) * previous_remains, cudaMemcpyDeviceToDevice);
            cudaMemcpy(previous, queries, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);

            cudaFree(distanceBuffer);
            cudaFree(idxBuffer);
            });
        timeParam["FilteringTime"] += TimeFilteringStep;
        if (reduced_size == 0) {
                auto TimeComputing = SPIN::TimeCheck([&]() {
                    size_t maxIDX;
                    float tmp;
                    //std::cout << eps << " " << previous_remains << std::endl;

                    //eps -= sqrt(3);

                    nnparam.query = pprevious;
                    nnparam.querySize = pprevious_remains;

                    float* distanceBuffer;
                    uint* idxBuffer;
                    cudaMalloc(&distanceBuffer, sizeof(float) * pprevious_remains);
                    cudaMalloc(&idxBuffer, sizeof(uint) * pprevious_remains);

                    //std::cout << "Remains : " << remains << ", 2step-previous :" << pprevious_remains << std::endl;

                    genAABBs(gpuRepresentative, eps, target.representative.size(), aabbBuffer);
                    cudaDeviceSynchronize();

                    targetGAS.refit(aabbBuffer, target.representative.size());

                    nnparam.traversable = targetGAS.handle;
                    nnparam.targetEPS = eps;

                    nnparam.distance = distanceBuffer;
                    nnparam.targetIDX = idxBuffer;
                    nnparam.state = COMPUTING;

                    launchParamBuffer.upload(&nnparam, 1);
                    int sqrtSize = ceil(sqrt(pprevious_remains));

                    auto traverseTime = SPIN::TimeCheck([&]() {
                        OPTIX_CHECK(optixLaunch(program.pipeline, 0,
                            launchParamBuffer.d_pointer(),
                            launchParamBuffer.sizeInBytes,
                            &program.sbt,
                            sqrtSize,
                            sqrtSize,
                            1
                        ));
                        cudaStreamSynchronize(0);
                        });
                    //std::cout << "Traverse: " << traverseTime << "ms" << std::endl;

                    tmp = getMaximumF(nnparam.distance, pprevious_remains, maxIDX);
                    uint targetIDX;
                    cudaMemcpy(&targetIDX, nnparam.targetIDX + maxIDX, sizeof(uint), cudaMemcpyDeviceToHost);
                    //std::cout << eps << " " << tmp << std::endl;

                    hDist = fmaxf(tmp, hDist);
                    cudaMemcpy(&cand1, pprevious + maxIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost);
                    cand1 = cand1 * voxelSize + aabb2min(targetAABB);
                    cudaMemcpy(&cand2, target.points + targetIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost);
                    //std::cout << length(cand1 - cand2) << std::endl;

                    cudaFree(distanceBuffer);
                    cudaFree(idxBuffer);
                    });
                timeParam["ComputingTime"] += TimeComputing;
        }
        else {
            cudaMemcpy(queries, reduced, sizeof(float3) * reduced_size, cudaMemcpyDeviceToDevice);
            //std::cout << "Hit size : " << remains - reduced_size << " / " << remains << std::endl;
        }
        pprevious_remains = previous_remains;
        previous_remains = remains;
        remains = reduced_size;

        cudaFree(reduced);
    }

    cudaFree(aabbBuffer);
    cudaFree(queries);
    cudaFree(gpuRepresentative);
    cudaFree(gpuClusterInfo);
    cudaFree(previous);
    cudaFree(pprevious);
    //cudaFree(culledA);
    launchParamBuffer.free();

    return hDist;
}

void upload(std::vector<float3>& HostData, HDGPUParam<HDMODE::POINT>& DeviceData) {
    cudaMemcpy(DeviceData.vert, HostData.data(), sizeof(float3) * HostData.size(), cudaMemcpyHostToDevice);
    DeviceData.vSize = HostData.size();
}

void download(HDGPUParam<HDMODE::POINT>& DeviceData, std::vector<float3>& HostData) {
    HostData.resize(DeviceData.vSize);
    cudaMemcpy(HostData.data(), DeviceData.vert, sizeof(float3) * DeviceData.vSize, cudaMemcpyDeviceToHost);
}

void HDGPUParam<HDMODE::POINT>::samplingVTX(std::vector<float3>& vtx, std::vector<uint3>& tris){
    vSize = vtx.size() + tris.size();
    cudaMalloc(&vert, sizeof(float3) * vSize);
    cudaMemcpy(vert, vtx.data(), sizeof(float3) * vtx.size(), cudaMemcpyHostToDevice);
    genCenterPointsOfTris(vert + vtx.size(), tris.size(), vert, tris);
}
 