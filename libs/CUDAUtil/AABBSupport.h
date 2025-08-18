#pragma once
#include "3rdParty/optix7support.h"
#include "3rdParty/helper_math.h"

__host__ __device__ inline
float3 aabb2min(OptixAabb aabb) {
    return make_float3(aabb.minX, aabb.minY, aabb.minZ);
}

__host__ __device__ inline
float3 aabb2max(OptixAabb aabb) {
    return make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);
}

__host__ __device__ inline
float3 aabb2center(OptixAabb aabb) {
    return (aabb2max(aabb) + aabb2min(aabb)) * 0.5f;
}

__host__ __device__ inline
float3 aabb2size(OptixAabb aabb) {
    return make_float3(aabb.maxX - aabb.minX, aabb.maxY - aabb.minY, aabb.maxZ - aabb.minZ);
}

__host__ __device__ inline
OptixAabb merge(const OptixAabb& a, const OptixAabb& b) {
    return { fminf(a.minX, b.minX),
        fminf(a.minY, b.minY),
        fminf(a.minZ, b.minZ),
        fmaxf(a.maxX, b.maxX),
        fmaxf(a.maxY, b.maxY),
        fmaxf(a.maxZ, b.maxZ) };
}

__host__ __device__ inline
OptixAabb intersection(const OptixAabb& a, const OptixAabb& b) {
    return { fmaxf(a.minX, b.minX),
        fmaxf(a.minY, b.minY),
        fmaxf(a.minZ, b.minZ),
        fminf(a.maxX, b.maxX),
        fminf(a.maxY, b.maxY),
        fminf(a.maxZ, b.maxZ) };
}

__host__ __device__ inline
float3 distance(const OptixAabb& a, const OptixAabb& b) {
    OptixAabb intersect = intersection(a, b);

    float3 res = { 0,0,0 };
    if (intersect.minX > intersect.maxX) {
        res.x = intersect.minX - intersect.maxX;
    }
    if (intersect.minY > intersect.maxY) {
        res.y = intersect.minY - intersect.maxY;
    }
    if (intersect.minZ > intersect.maxZ) {
        res.z = intersect.minZ - intersect.maxZ;
    }
    return res;
}
