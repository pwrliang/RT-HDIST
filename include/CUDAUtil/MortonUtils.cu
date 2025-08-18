#include "MortonUtils.h"

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_allocator.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/unique.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/distance.h>
#include <thrust/binary_search.h>
#include <thrust/remove.h>

#include <map>
#include <unordered_map>
#include "AABBSupport.h"

inline __host__ __device__ const float3 size(const OptixAabb& aabb) {
	return make_float3(aabb.maxX - aabb.minX, aabb.maxY - aabb.minY, aabb.maxZ - aabb.minZ);
}

inline __host__ __device__ const float3 floor(const float3& v) {
	return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

struct float3_min {
	__host__ __device__
		float3 operator()(float3 l, float3 r) {
		return fminf(l, r);
	}
};

struct float3_max {
	__host__ __device__
		float3 operator()(float3 l, float3 r) {
		return fmaxf(l, r);
	}
};

OptixAabb computeAABB_device(float3* vert, size_t vertSize) {
	float3 bmin = thrust::reduce(thrust::device, vert, vert + vertSize, make_float3(1e8f, 1e8f, 1e8f), float3_min());
	float3 bmax = thrust::reduce(thrust::device, vert, vert + vertSize, make_float3(-1e8f, -1e8f, -1e8f), float3_max());
	CUDA_SYNC_CHECK();

	return { bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z };
}

inline __host__ __device__ const uint64_t packing(const uint3 xyz, const BYTE bitCount) {
	uint64_t result = 0;

	for (uint bit = 0; bit < bitCount; ++bit) {
		result |= ((xyz.x & (1 << bit)) << (2 * bit)) |
			((xyz.y & (1 << bit)) << (2 * bit + 1)) |
			((xyz.z & (1 << bit)) << (2 * bit + 2));
	}

	return result;
}

inline __host__ __device__ uint3 unpacking(const uint64_t packed, const BYTE bitCount) {
	uint3 xyz = { 0, 0, 0 };

	for (uint bit = 0; bit < bitCount; ++bit) {
		// Extract bit for x, y, and z from packed result
		xyz.x |= ((packed >> (3 * bit)) & 1) << bit;
		xyz.y |= ((packed >> (3 * bit + 1)) & 1) << bit;
		xyz.z |= ((packed >> (3 * bit + 2)) & 1) << bit;
	}

	return xyz;
}

__global__ void _kernel_morton_code_aabb_bounds_(float3* vertices, uint64_t* encodes, size_t vertices_size, OptixAabb bounds, BYTE bitCount = 8) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tIdx >= vertices_size)
		return;

	const float3 voxelS = size(bounds) / (1 << bitCount);
	float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
	const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

	const float3 boundMin = make_float3(bounds.minX, bounds.minY, bounds.minZ);

	uint3 voxelIdx = make_uint3(floor((vertices[tIdx] - boundMin) / voxelSize));
	encodes[tIdx] = packing(voxelIdx, bitCount);
}

__global__ void _kernel_generate_aabb_with_morton_code__(uint64_t* encodes, OptixAabb* aabbs, float3* voxelIDs, size_t code_size, OptixAabb bounds, BYTE bitCount = 8) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (tIdx >= code_size)
		return;

	const float3 voxelS = size(bounds) / (1 << bitCount);
	float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
	const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

	const float3 boundMin = make_float3(bounds.minX, bounds.minY, bounds.minZ);
	float3 unpacked = make_float3(unpacking(encodes[tIdx], bitCount));

	float3 boxMin = boundMin + unpacked * voxelSize;
	float3 boxMax = boundMin + (unpacked + 1) * voxelSize;

	OptixAabb tAabb = { boxMin.x, boxMin.y, boxMin.z, boxMax.x, boxMax.y, boxMax.z };
	aabbs[tIdx] = tAabb;
	voxelIDs[tIdx] = unpacked;
}

void genUniformClusterAabb(float3* d_points, size_t pointSize, OptixAabb*& cluster, float3*& voxelIDs, size_t& number_of_cluster, std::vector<size_t>& vertex_pos_per_cluster, BYTE bitCount) {
	uint64_t* mortons;
	cudaMalloc(&mortons, pointSize * sizeof(uint64_t));

	OptixAabb _bound = computeAABB_device(d_points, pointSize);

	dim3 grids(ceil((pointSize + 1) / 1024.), 1);
	_kernel_morton_code_aabb_bounds_ << <grids, 1024 >> > (d_points, mortons, pointSize, _bound, bitCount);

	thrust::sort_by_key(thrust::device, mortons, mortons + pointSize, d_points);

	thrust::device_vector<uint64_t> clusterID(pointSize);
	thrust::device_vector<uint64_t> d_count(pointSize);

	auto newEnd = thrust::reduce_by_key(thrust::device, mortons, mortons + pointSize, thrust::constant_iterator<int>(1), clusterID.begin(), d_count.begin());
	number_of_cluster = newEnd.first - clusterID.begin();

	clusterID.resize(number_of_cluster);
	d_count.resize(number_of_cluster);

	thrust::device_vector<int> d_reduced_ptr(number_of_cluster + 1);
	thrust::exclusive_scan(d_count.begin(), d_count.end(), d_reduced_ptr.begin());
	d_reduced_ptr[number_of_cluster] = pointSize;

	vertex_pos_per_cluster.resize(number_of_cluster + 1);
	thrust::copy(d_reduced_ptr.begin(), d_reduced_ptr.end(), vertex_pos_per_cluster.begin());

	cudaMalloc(&cluster, sizeof(OptixAabb) * number_of_cluster);
	cudaMalloc(&voxelIDs, sizeof(float3) * number_of_cluster);

	grids = dim3(ceil((number_of_cluster + 1) / 1024.), 1);
	_kernel_generate_aabb_with_morton_code__ << <grids, 1024 >> > (thrust::raw_pointer_cast(&clusterID[0]), cluster, voxelIDs, number_of_cluster, _bound, bitCount);

	cudaFree(mortons);
	cudaDeviceSynchronize();
}


void genUniformClusterAabb(float3* d_points, size_t pointSize, uint3* tris, OptixAabb*& cluster, float3*& voxelIDs, size_t& number_of_cluster, std::vector<size_t>& vertex_pos_per_cluster, BYTE bitCount) {
	uint64_t* mortons;
	cudaMalloc(&mortons, pointSize * sizeof(uint64_t));

	OptixAabb _bound = computeAABB_device(d_points, pointSize);

	dim3 grids(ceil((pointSize + 1) / 1024.), 1);
	_kernel_morton_code_aabb_bounds_ << <grids, 1024 >> > (d_points, mortons, pointSize, _bound, bitCount);

	thrust::sort_by_key(thrust::device, mortons, mortons + pointSize, d_points);
	thrust::sort_by_key(thrust::device, mortons, mortons + pointSize, tris);

	thrust::device_vector<uint64_t> clusterID(pointSize);
	thrust::device_vector<uint64_t> d_count(pointSize);

	auto newEnd = thrust::reduce_by_key(thrust::device, mortons, mortons + pointSize, thrust::constant_iterator<int>(1), clusterID.begin(), d_count.begin());
	number_of_cluster = newEnd.first - clusterID.begin();

	clusterID.resize(number_of_cluster);
	d_count.resize(number_of_cluster);

	thrust::device_vector<int> d_reduced_ptr(number_of_cluster + 1);
	thrust::exclusive_scan(d_count.begin(), d_count.end(), d_reduced_ptr.begin());
	d_reduced_ptr[number_of_cluster] = pointSize;

	vertex_pos_per_cluster.resize(number_of_cluster + 1);
	thrust::copy(d_reduced_ptr.begin(), d_reduced_ptr.end(), vertex_pos_per_cluster.begin());

	cudaMalloc(&cluster, sizeof(OptixAabb) * number_of_cluster);
	cudaMalloc(&voxelIDs, sizeof(float3) * number_of_cluster);

	grids = dim3(ceil((number_of_cluster + 1) / 1024.), 1);
	_kernel_generate_aabb_with_morton_code__ << <grids, 1024 >> > (thrust::raw_pointer_cast(&clusterID[0]), cluster, voxelIDs, number_of_cluster, _bound, bitCount);

	cudaFree(mortons);
	cudaDeviceSynchronize();
}

__global__ void genAABBs_kernel_(float3* points, float radius, size_t numPoints, OptixAabb* d_aabb) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	while (tIdx < numPoints) {
		if (tIdx >= numPoints)
			return;
		float3 center = points[tIdx];

		float3 m_min = center - radius;
		float3 m_max = center + radius;

		d_aabb[tIdx] = { m_min.x, m_min.y, m_min.z,
						m_max.x, m_max.y, m_max.z };

		tIdx += 1024 * 1024;
	}
}


void genAABBs(float3* points, float radius, size_t numPoints, OptixAabb* d_aabb) {
	size_t threads = 1024;
	size_t blocks = 1024;

	genAABBs_kernel_ << <blocks, threads >> > (
		points,
		radius,
		numPoints,
		d_aabb
		);
}


__global__ void __kernel__center_point_generation__(float3* gpuBuff, size_t numTris, float3* vtxBuff, uint3* triBuff) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	uint3 triIDX;
	float3 vtx;

	while (tIdx < numTris) {
		if (tIdx >= numTris)
			return;

		triIDX = triBuff[tIdx];
		vtx = vtxBuff[triIDX.x];
		vtx += vtxBuff[triIDX.y];
		vtx += vtxBuff[triIDX.z];
		vtx *= 1.0f/3.0f;

		gpuBuff[tIdx] = vtx;

		tIdx += 1024 * 1024;
	}
}

void genCenterPointsOfTris(float3* gpuBuff, size_t numTris, float3* vtxBuff, std::vector<uint3>& tris){
	CUDABuffer indexBuff;
	indexBuff.alloc_and_upload(tris);

	size_t threads = 1024;
	size_t blocks = 1024;

	__kernel__center_point_generation__<<<blocks, threads>>>(gpuBuff, numTris, vtxBuff, (uint3*)indexBuff.d_pointer());

	indexBuff.free();
}


void genCenterPointsOfTris(float3*& gpuBuff, size_t numTris, float3* vtxBuff, uint3* trisGPU) {
	size_t threads = 1024;
	size_t blocks = 1024;

	__kernel__center_point_generation__ << <blocks, threads >> > (gpuBuff, numTris, vtxBuff, trisGPU);
}

__global__ void __kernel__proj_to_index_space__(float3* vtx, size_t numVtx, float3* indexs, OptixAabb bounds, float3 voxelSize) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const float3 boundMin = { bounds.minX, bounds.minY, bounds.minZ };
	
	float3 _vtx;

	while (tIdx < numVtx) {
		if (tIdx >= numVtx)
			return;

		_vtx = vtx[tIdx];
		_vtx = floor((_vtx - boundMin) / voxelSize);

		
		indexs[tIdx] = _vtx;

		tIdx += 1024 * 1024;
	}

}

void projToIndexSpace(float3* vtx, size_t numVtx, float3* onIndexSpace, OptixAabb bounds, float3 voxelSize){
	size_t threads = 1024;
	size_t blocks = 1024;

	__kernel__proj_to_index_space__ << <blocks, threads >> > (vtx, numVtx, onIndexSpace, bounds, voxelSize);
}

struct float3_compare {
	__host__ __device__
		bool operator()(const float3& a, const float3& b) const {
		if (a.x != b.x) return a.x < b.x;
		if (a.y != b.y) return a.y < b.y;
		return a.z < b.z;
	}
};

// Custom equality for float3
struct float3_equal {
	__host__ __device__
		bool operator()(const float3& a, const float3& b) const {
		return a.x == b.x && a.y == b.y && a.z == b.z;
	}
};


void reduceAndGetLUT(float3* vtx, size_t numVtx, float3** onIndexSpace, uint** LUT, uint** idxIDXquery,size_t &uniqueSize,OptixAabb bounds, float3 voxelSize){
	cudaMalloc(onIndexSpace, sizeof(float3) * numVtx);

	projToIndexSpace(vtx, numVtx, *onIndexSpace, bounds, voxelSize);

	thrust::sort_by_key(thrust::device, *onIndexSpace, *onIndexSpace + numVtx, vtx, float3_compare());

	thrust::device_vector<uint64_t> d_count(numVtx);
	auto end = thrust::reduce_by_key(thrust::device, *onIndexSpace, *onIndexSpace + numVtx, thrust::constant_iterator<int>(1), *onIndexSpace, d_count.begin(), float3_equal());
	uniqueSize = end.first - *onIndexSpace;
	//idx.resize(uniqueSize);

	d_count.resize(uniqueSize);

	thrust::device_vector<int> d_reduced_ptr(uniqueSize + 1);
	thrust::exclusive_scan(d_count.begin(), d_count.end(), d_reduced_ptr.begin());
	d_reduced_ptr[uniqueSize] = numVtx;

	cudaMalloc(LUT, sizeof(uint) * (uniqueSize+1));
	cudaMemcpy(*LUT, thrust::raw_pointer_cast(d_reduced_ptr.data()), sizeof(uint) * (uniqueSize + 1), cudaMemcpyDeviceToDevice);

	cudaMalloc(idxIDXquery, sizeof(uint) * uniqueSize);
	thrust::sequence(thrust::device, *idxIDXquery, *idxIDXquery + uniqueSize);
}

__global__ void __kernel__to_index_space__(float3* vtx, size_t numVtx, float3* indexs, OptixAabb bounds, float3 voxelSize) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const float3 boundMin = { bounds.minX, bounds.minY, bounds.minZ };

	float3 _vtx;

	while (tIdx < numVtx) {
		if (tIdx >= numVtx)
			return;

		_vtx = vtx[tIdx];
		_vtx = ((_vtx - boundMin) / voxelSize);


		indexs[tIdx] = _vtx;

		tIdx += 1024 * 1024;
	}

}

void ToIndexSpace(float3* vtx, size_t numVtx, float3* onIndexSpace, OptixAabb bounds, float3 voxelSize) {
	size_t threads = 1024;
	size_t blocks = 1024;

	__kernel__to_index_space__ << <blocks, threads >> > (vtx, numVtx, onIndexSpace, bounds, voxelSize);
}

inline __host__ __device__ bool testAABBPlane(const float3 boxMin, const float3 boxMax, const float3 normal, const float denom) {
	float3 c = (boxMax + boxMin) * 0.5f;
	float3 e = boxMax - c;

	float r = e.x * abs(normal.x) + e.y * abs(normal.y) + e.z * abs(normal.z);
	float s = dot(normal, c) - denom;

	return abs(s) <= r;
}

inline __host__ __device__ bool triangleBoxIntersect(
	const float3 v0, const float3 v1, const float3 v2,
	const float3 boxCenter, const float3 boxHalfSize)
{
	float3 tv0 = v0 - boxCenter;
	float3 tv1 = v1 - boxCenter;
	float3 tv2 = v2 - boxCenter;

	float3 e0 = tv1 - tv0;
	float3 e1 = tv2 - tv1;
	float3 e2 = tv0 - tv2;

	float3 edges[3] = { e0, e1, e2 };

	float3 fex = fabs(edges[0]);
	float3 fey = fabs(edges[1]);
	float3 fez = fabs(edges[2]);

#define AXISTEST_X01(a, b, fa, fb) {float p0 = a * tv0.y - b * tv0.z; float p2 = a * tv2.y - b * tv2.z; float min_p = fminf(p0, p2); float max_p = fmaxf(p0, p2); float rad = fa * boxHalfSize.y + fb * boxHalfSize.z; if (min_p > rad || max_p < -rad) return false;}
#define AXISTEST_Y02(a, b, fa, fb) {float p0 = -a * tv0.x + b * tv0.z; float p2 = -a * tv2.x + b * tv2.z; float min_p = fminf(p0, p2); float max_p = fmaxf(p0, p2); float rad = fa * boxHalfSize.x + fb * boxHalfSize.z; if (min_p > rad || max_p < -rad) return false;}
#define AXISTEST_Z12(a, b, fa, fb) {float p1 = a * tv1.x - b * tv1.y; float p2 = a * tv2.x - b * tv2.y; float min_p = fminf(p1, p2); float max_p = fmaxf(p1, p2); float rad = fa * boxHalfSize.x + fb * boxHalfSize.y; if (min_p > rad || max_p < -rad) return false;}

	AXISTEST_X01(e0.z, e0.y, fex.z, fex.y);
	AXISTEST_Y02(e0.z, e0.x, fex.z, fex.x);
	AXISTEST_Z12(e0.y, e0.x, fex.y, fex.x);

	AXISTEST_X01(e1.z, e1.y, fey.z, fey.y);
	AXISTEST_Y02(e1.z, e1.x, fey.z, fey.x);
	AXISTEST_Z12(e1.y, e1.x, fey.y, fey.x);

	AXISTEST_X01(e2.z, e2.y, fez.z, fez.y);
	AXISTEST_Y02(e2.z, e2.x, fez.z, fez.x);
	AXISTEST_Z12(e2.y, e2.x, fez.y, fez.x);

#undef AXISTEST_X01
#undef AXISTEST_Y02
#undef AXISTEST_Z12

	// Test AABB against triangle bbox
	float3 triMin = fminf(fminf(tv0, tv1), tv2);
	float3 triMax = fmaxf(fmaxf(tv0, tv1), tv2);

	if (triMin.x > boxHalfSize.x || triMax.x < -boxHalfSize.x) return false;
	if (triMin.y > boxHalfSize.y || triMax.y < -boxHalfSize.y) return false;
	if (triMin.z > boxHalfSize.z || triMax.z < -boxHalfSize.z) return false;

	// Test if the box intersects the plane of the triangle
	float3 normal = cross(e0, e1);
	float d = -(normal.x * tv0.x + normal.y * tv0.y + normal.z * tv0.z);

	float _boxHalfSize[3] = { boxHalfSize.x, boxHalfSize.y, boxHalfSize.z };
	float _normal[3] = { normal.x, normal.y, normal.z };
	float _vmin[3], _vmax[3];
	for (int q = 0; q < 3; q++) {
		if (_normal[q] > 0.0f) {
			_vmin[q] = -_boxHalfSize[q];
			_vmax[q] = _boxHalfSize[q];
		}
		else {
			_vmin[q] = _boxHalfSize[q];
			_vmax[q] = -_boxHalfSize[q];
		}
	}
	float3 vmin = { _vmin[0], _vmin[1], _vmin[2] };
	float3 vmax = { _vmax[0], _vmax[1], _vmax[2] };

	if ((normal.x * vmin.x + normal.y * vmin.y + normal.z * vmin.z + d) > 0.0f) return false;
	if ((normal.x * vmax.x + normal.y * vmax.y + normal.z * vmax.z + d) < 0.0f) return false;

	return true;
}


void voxelize_cpu(float3* vtx, size_t numVtx, uint3* idx, size_t numTris, float3*& voxelIDs, size_t& num_of_cluster, uint*& idxList, std::vector<size_t>& index_pos_per_cluster, BYTE bitCount) {
	std::vector<float3> vtxHost(numVtx);
	std::vector<uint3> idxHost(numTris);

	std::unordered_map<uint64_t, std::vector<uint>> voxelTriTable;

	OptixAabb _bound = computeAABB_device(vtx, numVtx);
	cudaMemcpy(vtxHost.data(), vtx, sizeof(float3) * numVtx, cudaMemcpyDeviceToHost);
	cudaMemcpy(idxHost.data(), idx, sizeof(uint3) * numTris, cudaMemcpyDeviceToHost);

	const float3 voxelS = size(_bound) / (1 << bitCount);
	float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
	const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

	std::cout << delimiter << std::endl;

	const float3 boundMin = make_float3(_bound.minX, _bound.minY, _bound.minZ);

	size_t triIDX = 0;
	for (auto& tris : idxHost) {
		float3 v0 = vtxHost[tris.x];
		float3 v1 = vtxHost[tris.y];
		float3 v2 = vtxHost[tris.z];

		uint3 vIdx0 = make_uint3(floor((v0 - boundMin) / voxelSize));
		uint3 vIdx1 = make_uint3(floor((v1 - boundMin) / voxelSize));
		uint3 vIdx2 = make_uint3(floor((v2 - boundMin) / voxelSize));

		uint3 minVoxel = min(min(vIdx0, vIdx1), vIdx2);
		uint3 maxVoxel = max(max(vIdx0, vIdx1), vIdx2) + 1;

		for (uint zz = minVoxel.z; zz <= maxVoxel.z; zz++) {
			for (uint yy = minVoxel.y; yy <= maxVoxel.y; yy++) {
				for (uint xx = minVoxel.x; xx <= maxVoxel.x; xx++) {
					float3 voxelMin = make_float3(xx, yy, zz) * voxelSize + boundMin;
					float3 voxelMax = voxelMin + voxelSize;

					//triangle-box intersection
					//intersection(v0,v1,v2, voxelMin, voxelMax)
					bool isIntersected = triangleBoxIntersect(v0, v1, v2, (voxelMin+voxelMax)*0.5f, (voxelMax-voxelMin)*0.5f);
					if (isIntersected) {
						uint64_t voxelID = packing({ xx,yy,zz }, bitCount);
						voxelTriTable[voxelID].push_back(triIDX);
					}
				}
			}
		}
		triIDX++;
	}

	std::vector<float3> voxelList;
	std::vector<size_t> voxelTriLB;
	std::vector<uint> voxelTriList;

	voxelTriLB.push_back(0);
	size_t exclusive_sum = 0;
	//Serialize
	for (auto& data : voxelTriTable) {
		uint3 rep = unpacking(data.first, bitCount);
		voxelList.push_back(make_float3(rep));
		exclusive_sum += data.second.size();
		voxelTriLB.push_back(exclusive_sum);
		voxelTriList.insert(voxelTriList.end(), data.second.begin(), data.second.end());
	}

	num_of_cluster = voxelList.size();
	cudaMalloc(&voxelIDs, sizeof(float3) * num_of_cluster);
	cudaMalloc(&idxList, sizeof(uint) * exclusive_sum);

	cudaMemcpy(voxelIDs, voxelList.data(), sizeof(float3) * voxelList.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(idxList, voxelTriList.data(), sizeof(uint) * exclusive_sum, cudaMemcpyHostToDevice);
	index_pos_per_cluster = voxelTriLB;
}

__global__ void __kernel_compute_triangles_per_voxel__(float3* vtx, size_t numVtx, uint3* tris, size_t numTris, uint* number_of_tris, OptixAabb bounds, float3 voxelSize, int gridSIze) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	const float3 boundMin = make_float3(bounds.minX, bounds.minY, bounds.minZ);
	
	uint3 tri;
	float3 v0, v1, v2;
	uint3 vIdx0, vIdx1, vIdx2;
	uint3 minVoxel, maxVoxel;

	while (tIdx < numTris) {
		tri = tris[tIdx];
		v0 = vtx[tri.x];
		v1 = vtx[tri.y];
		v2 = vtx[tri.z];

		vIdx0 = make_uint3(floor((v0 - boundMin) / voxelSize));
		vIdx1 = make_uint3(floor((v1 - boundMin) / voxelSize));
		vIdx2 = make_uint3(floor((v2 - boundMin) / voxelSize));

		minVoxel = min(min(vIdx0, vIdx1), vIdx2);
		maxVoxel = max(max(vIdx0, vIdx1), vIdx2) + 1;
		
		for (uint zz = minVoxel.z; zz <= maxVoxel.z; zz++) {
			for (uint yy = minVoxel.y; yy <= maxVoxel.y; yy++) {
				for (uint xx = minVoxel.x; xx <= maxVoxel.x; xx++) {
					float3 voxelMin = make_float3(xx, yy, zz) * voxelSize + boundMin;
					float3 voxelMax = voxelMin + voxelSize;

					//triangle-box intersection
					//intersection(v0,v1,v2, voxelMin, voxelMax)
					bool isIntersected = triangleBoxIntersect(v0, v1, v2, (voxelMin + voxelMax) * 0.5f, (voxelMax - voxelMin) * 0.5f);
					if (isIntersected) {
						size_t voxelID = zz * gridSIze * gridSIze + yy * gridSIze + xx;
						atomicAdd(&number_of_tris[voxelID], 1);
					}
				}
			}
		}

		tIdx += 1024 * 1024;
	}
}

void voxelize_gpu(float3* vtx, size_t numVtx, uint3* idx, size_t numTris, float3*& voxelIDs, size_t& num_of_cluster, uint*& idxList, std::vector<size_t>& index_pos_per_cluster, BYTE bitCount) {
	OptixAabb _bound = computeAABB_device(vtx, numVtx);

	const float3 voxelS = size(_bound) / (1 << bitCount);
	float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
	const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

	const float3 boundMin = make_float3(_bound.minX, _bound.minY, _bound.minZ);

	size_t maxVoxel = 1ULL << (3 * bitCount);
	size_t gridSize = 1ULL << bitCount;

	uint* count;
	cudaMalloc(&count, sizeof(uint)*maxVoxel);

	size_t threads = 1024;
	size_t blocks = 1024;
	__kernel_compute_triangles_per_voxel__ << <blocks, threads >> > (vtx, numVtx, idx, numTris, count, _bound, voxelSize, gridSize);

	uint* voxelID;
	cudaMalloc(&voxelID, sizeof(uint) * maxVoxel);
	thrust::sequence(thrust::device, voxelID, voxelID + maxVoxel);


	//auto last = thrust::copy_if(thrust::device, voxelID, voxelID + maxVoxel, voxelID, [](uint x) {return x > 0;});


	cudaFree(count);
}

struct is_in_rmKeys {
	thrust::device_ptr<const uint64_t> begin;
	thrust::device_ptr<const uint64_t> end;

	is_in_rmKeys(thrust::device_ptr<const uint64_t> b, thrust::device_ptr<const uint64_t> e)
		: begin(b), end(e) {}

	__host__ __device__
		bool operator()(const thrust::tuple<uint64_t, float3>& t) const {
		uint64_t key = thrust::get<0>(t);
		return thrust::binary_search(thrust::seq, begin, end, key);
	}
};

void overlapCulling(float3* vertA, size_t sizeA, float3* vertB, size_t sizeB, float3*& res, size_t &sizeRes, BYTE bitCount) {
	OptixAabb boundA = computeAABB_device(vertA, sizeA);
	OptixAabb boundB = computeAABB_device(vertB, sizeB);

	OptixAabb totalAABB = merge(boundA, boundB);

	const float3 voxelS = (aabb2max(totalAABB) - aabb2min(totalAABB)) / (1ULL << bitCount);
	float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
	const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

	uint64_t* rmKeys;
	cudaMalloc(&rmKeys, sizeB * sizeof(uint64_t));

	dim3 grids(ceil((sizeB + 1) / 1024.), 1);
	_kernel_morton_code_aabb_bounds_ << <grids, 1024 >> > (vertB, rmKeys, sizeB, totalAABB, bitCount);

	thrust::device_ptr<uint64_t> rmKeysBegin = thrust::device_pointer_cast(rmKeys);
	thrust::sort(thrust::device, rmKeysBegin, rmKeysBegin + sizeB);
	auto rmKeysEnd = thrust::unique(thrust::device, rmKeysBegin, rmKeysBegin + sizeB);
	int uniqueRmSize = rmKeysEnd - rmKeysBegin;

	uint64_t* Akeys;
	cudaMalloc(&Akeys, sizeA * sizeof(uint64_t));

	grids = dim3(ceil((sizeA + 1) / 1024.), 1);
	_kernel_morton_code_aabb_bounds_ << <grids, 1024 >> > (vertA, Akeys, sizeA, totalAABB, bitCount);

	//thrust::device_ptr<uint64_t> AKeysBegin = thrust::device_pointer_cast(Akeys);
	//thrust::sort(thrust::device, AKeysBegin, AKeysBegin + sizeA);
	//auto AKeysEnd = thrust::unique(thrust::device, AKeysBegin, AKeysBegin + sizeA);
	//int uniqueASize = AKeysEnd - AKeysBegin;

	//std::cout << uniqueASize << " " << uniqueRmSize << std::endl;

	// ----- Wrap Akeys and vertA as device_vector for Thrust
	thrust::device_ptr<uint64_t> AkeysBegin = thrust::device_pointer_cast(Akeys);
	thrust::device_vector<uint64_t> d_Akeys(AkeysBegin, AkeysBegin + sizeA);
	thrust::device_vector<float3> d_vertA(vertA, vertA + sizeA);

	// ----- Zip Akeys and vertA together
	auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(d_Akeys.begin(), d_vertA.begin()));
	auto zipped_end = thrust::make_zip_iterator(thrust::make_tuple(d_Akeys.end(), d_vertA.end()));

	// ----- Remove elements in A where key is in rmKeys
	auto new_end = thrust::remove_if(
		zipped_begin, zipped_end,
		is_in_rmKeys(rmKeysBegin, rmKeysEnd)
	);

	// ----- Compute new size
	size_t newSize = new_end - zipped_begin;

	// Resize vectors
	d_Akeys.resize(newSize);
	d_vertA.resize(newSize);

	// ----- Output result
	cudaMalloc(&res, newSize * sizeof(float3));
	cudaMemcpy(res, thrust::raw_pointer_cast(d_vertA.data()), newSize * sizeof(float3), cudaMemcpyDeviceToDevice);
	sizeRes = newSize;

	// Cleanup
	cudaFree(rmKeys);
	cudaFree(Akeys);
}