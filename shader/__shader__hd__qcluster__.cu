#include "OptiXGlobalHelper.h"
#include "params.h"

extern "C" __constant__ OptiXHDistParam optixLaunchParams;

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

struct Payload_t {
    float3 originVtx;
    float minDist;
    float rawMinDist;
    //float3 scaledQuery;
    uint targetIdx;

    bool terminated;
};

inline __host__ __device__ const float3 floor(const float3& v) {
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

extern "C" __global__ void __raygen__program__() {
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int idx = iy * optixGetLaunchDimensions().x + ix;

    if (idx >= optixLaunchParams.querySize) return;

    //float3 boundMin = { optixLaunchParams.targetBound.minX, optixLaunchParams.targetBound.minY, optixLaunchParams.targetBound.minZ };
    //float3 boundMax = { optixLaunchParams.targetBound.maxX, optixLaunchParams.targetBound.maxY, optixLaunchParams.targetBound.maxZ };
    //float3 projQuery = clamp(optixLaunchParams.query[idx], boundMin, boundMax);

    //float3 rayOrigin = ((optixLaunchParams.query[idx] - boundMin) / optixLaunchParams.targetVoxelSize);
    float3 rayOrigin = optixLaunchParams.query[idx];
    //float3 scaledQuery = ((optixLaunchParams.query[idx] - boundMin) / optixLaunchParams.targetVoxelSize);
    //float3 rayOrigin = floor((projQuery - boundMin)/optixLaunchParams.targetVoxelSize);

    float3 rayDir = normalize(make_float3(1, 0, 0));

    float tmin = 1e-8f;
    float tmax = 1e-7f;


    Payload_t payload;
    //payload.scaledQuery = scaledQuery;
    payload.minDist = 1e8f;
    payload.rawMinDist = 1e8f;
    payload.targetIdx = -1;
    payload.terminated = false;

    if (optixLaunchParams.state == COMPUTING) {
        float3 boundMin = { optixLaunchParams.targetBound.minX, optixLaunchParams.targetBound.minY, optixLaunchParams.targetBound.minZ };
        float3 originVtx = rayOrigin * optixLaunchParams.targetVoxelSize + boundMin;
        payload.originVtx = originVtx;
    }

    uint32_t u0, u1;
    packPointer(&payload, u0, u1);

    optixTrace(optixLaunchParams.traversable,
        rayOrigin,
        rayDir,
        tmin,    // tmin
        tmax,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE, //OPTIX_RAY_FLAG_NONE,
        SURFACE_RAY_TYPE,             // SBT offset
        RAY_TYPE_COUNT,               // SBT stride
        SURFACE_RAY_TYPE,             // missSBTIndex 
        u0, u1);

    if (payload.rawMinDist < 1e8f) {
        optixLaunchParams.targetIDX[idx] = payload.targetIdx;
        optixLaunchParams.distance[idx] = payload.rawMinDist;
    }
    else {
        optixLaunchParams.targetIDX[idx] = -1;
        optixLaunchParams.distance[idx] = -1;
    }
}

extern "C" __global__ void __miss__radiance() {
}

extern "C" __global__ void __anyhit__radiance() {
    optixTerminateRay();
}

extern "C" __global__ void __intersection__radiance() {
    Payload_t& prd = *(Payload_t*)getPRD<Payload_t>();

    if (optixLaunchParams.state == FILTERING && prd.terminated) {
        return;
    }

    uint primIdx = optixGetPrimitiveIndex(); // voxel Primitive

    const float3 center = optixLaunchParams.representative[primIdx];
    const float3 aabbcenter = optixLaunchParams.representative[primIdx] + optixLaunchParams.targetTransform;
    const float3 rayOrigin = optixGetWorldRayOrigin();
    float dist = length(center - rayOrigin);
    float hitDist = length(aabbcenter - rayOrigin);

    if (hitDist < optixLaunchParams.targetEPS) { // condition of intersection

        if (optixLaunchParams.state == COMPUTING) {
            size_t last = optixLaunchParams.clusterInfo[primIdx + 1];

            for (int i = optixLaunchParams.clusterInfo[primIdx]; i < last; i++) {
                float rawdist = length(prd.originVtx - optixLaunchParams.target[i]);

                if (prd.rawMinDist >= rawdist) {
                    prd.rawMinDist = rawdist;
                    prd.targetIdx = i;
                }
            }
        }
        else {
            prd.rawMinDist = 1.0;
            prd.terminated = true;
            optixReportIntersection(2e-8f, 0);
        }
        //atomicMin(prd.distBPtr + primIdx, dist);
    }
}

extern "C" __global__ void __closesthit__radiance() {
}
