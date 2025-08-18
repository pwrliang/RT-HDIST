#include "ReduceUtils.h"

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

float getMaximumF(float* value, size_t size, size_t& idx) {
	float* ptr = thrust::max_element(thrust::device, value, value + size);
	idx = ptr - value;
	float temp;
	cudaMemcpy(&temp, ptr, sizeof(float) * 1, cudaMemcpyDeviceToHost);

	return temp;
}

float getMinimumF(float* value, size_t size, size_t& idx) {
	float* ptr = thrust::min_element(thrust::device, value, value + size);
	idx = ptr - value;
	float temp;
	cudaMemcpy(&temp, ptr, sizeof(float) * 1, cudaMemcpyDeviceToHost);

	return temp;
}

template <bool Cond>
struct kernel_distance_verifier;

template <>
struct kernel_distance_verifier<true> {
	__host__ __device__ bool operator()(float x) {
		return x > 0;
	}
};

template <>
struct kernel_distance_verifier<false> {
	__host__ __device__ bool operator()(float x) {
		return x <= 0;
	}
};

/// <summary>
/// 
/// </summary>
/// <param name="source">Source points</param>
/// <param name="source_size">size of source</param>
/// <param name="distance">reference distance</param>
/// <param name="result">reduced points</param>
/// <param name="new_size">size of new points</param>
/// <param name="op">binary operator true or false (check is hit with distance > 0)</param>
void reduceWithDistance(float3* source, size_t source_size, float* distance, float3*& result, size_t& new_size, const bool op) {
	float3* tempBuff;
	cudaMalloc(&tempBuff, sizeof(float3) * (source_size));

	float3* last;

	if (op)
		last = thrust::copy_if(thrust::device, source, source + source_size, distance, tempBuff, kernel_distance_verifier<true>());
	else
		last = thrust::copy_if(thrust::device, source, source + source_size, distance, tempBuff, kernel_distance_verifier<false>());

	new_size = last - tempBuff;

	cudaMalloc(&result, sizeof(float3) * new_size);
	cudaMemcpy(result, tempBuff, sizeof(float3) * new_size, cudaMemcpyDeviceToDevice);

	cudaFree(tempBuff);
}

/// <summary>
/// 
/// </summary>
/// <param name="source">Source points</param>
/// <param name="source_size">size of source</param>
/// <param name="distance">reference distance</param>
/// <param name="result">reduced points</param>
/// <param name="new_size">size of new points</param>
/// <param name="op">binary operator true or false (check is hit with distance > 0)</param>
void reduceWithDistance_2(float3* source, uint* idxs, size_t source_size, float* distance, float3*& result, uint*& idxsResult, size_t& new_size, const bool op) {
	float3* tempBuff;
	uint* tempIdxBuff;
	cudaMalloc(&tempBuff, sizeof(float3) * (source_size));
	cudaMalloc(&tempIdxBuff, sizeof(uint) * (source_size));

	float3* last;

	if (op) {
		last = thrust::copy_if(thrust::device, source, source + source_size, distance, tempBuff, kernel_distance_verifier<true>());
		thrust::copy_if(thrust::device, idxs, idxs + source_size, distance, tempIdxBuff, kernel_distance_verifier<true>());
	}
	else {
		last = thrust::copy_if(thrust::device, source, source + source_size, distance, tempBuff, kernel_distance_verifier<false>());
		thrust::copy_if(thrust::device, idxs, idxs + source_size, distance, tempIdxBuff, kernel_distance_verifier<false>());
	}

	new_size = last - tempBuff;

	cudaMalloc(&result, sizeof(float3) * new_size);
	cudaMemcpy(result, tempBuff, sizeof(float3) * new_size, cudaMemcpyDeviceToDevice);

	cudaMalloc(&idxsResult, sizeof(uint) * new_size);
	cudaMemcpy(idxsResult, tempIdxBuff, sizeof(uint) * new_size, cudaMemcpyDeviceToDevice);

	cudaFree(tempBuff);
	cudaFree(tempIdxBuff);
}

__global__ void __kernel__count_tris_per_points__(float3* vtxBuff, size_t numVtx, uint3* triBuff, size_t numTris, uint* resBuff) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	uint3 triIDX;

	while (tIdx < numTris) {
		if (tIdx >= numTris)
			return;

		triIDX = triBuff[tIdx];

		atomicAdd(&resBuff[triIDX.x], 1);
		atomicAdd(&resBuff[triIDX.y], 1);
		atomicAdd(&resBuff[triIDX.z], 1);

		tIdx += 1024 * 1024;
	}
}

__global__ void __kernel__register_tris_per_points__(float3* vtxBuff, size_t numVtx, uint3* triBuff, size_t numTris, uint* resBuff, uint* mapBuff, uint* refBuff) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	uint3 triIDX;
	uint idx;

	while (tIdx < numTris) {
		if (tIdx >= numTris)
			return;

		triIDX = triBuff[tIdx];

		idx = atomicAdd(&resBuff[triIDX.x], 1);
		mapBuff[refBuff[triIDX.x] + idx] = tIdx;
		idx = atomicAdd(&resBuff[triIDX.y], 1);
		mapBuff[refBuff[triIDX.y] + idx] = tIdx;
		idx = atomicAdd(&resBuff[triIDX.z], 1);
		mapBuff[refBuff[triIDX.z] + idx] = tIdx;

		tIdx += 1024 * 1024;
	}
}


void countTrianglesPerVertex(float3* vtx, size_t vtx_size, uint3* tris, size_t tri_size, uint* res) {
	cudaMalloc(&res, sizeof(uint) * vtx_size);

	size_t threads = 1024;
	size_t blocks = 1024;

	__kernel__count_tris_per_points__ << <blocks, threads >> > (vtx, vtx_size, tris, tri_size, res);
}

void makeVertexTrianlgeMap(float3* vtx, size_t vtx_size, uint3* tris, size_t tri_size, uint* res, uint* map, uint &map_size, uint* ref) {
	cudaMalloc(&res, sizeof(uint) * vtx_size);

	size_t threads = 1024;
	size_t blocks = 1024;

	__kernel__count_tris_per_points__ << <blocks, threads >> > (vtx, vtx_size, tris, tri_size, res);

	//map_size = thrust::reduce(thrust::device, res, res + vtx_size, 0, thrust::plus<uint>());

	cudaMalloc(&ref, sizeof(uint) * (vtx_size+1));
	thrust::exclusive_scan(thrust::device, res, res + vtx_size, ref);

	uint lastValue;
	uint totalSum;
	cudaMemcpy(&lastValue, res + vtx_size - 1, sizeof(uint), cudaMemcpyDeviceToHost); // Last element of the original array
	cudaMemcpy(&totalSum, ref + vtx_size - 1, sizeof(uint), cudaMemcpyDeviceToHost); // Last element of the scan result
	totalSum += lastValue; // Add the last value of the original array to the total sum
	cudaMemcpy(ref + vtx_size, &totalSum, sizeof(uint), cudaMemcpyHostToDevice);
	map_size = totalSum;

	uint* tempRes;
	cudaMalloc(&tempRes, sizeof(uint) * vtx_size);
	cudaMalloc(&map, sizeof(uint) * map_size);

	__kernel__register_tris_per_points__<<<1024, 1024>>>(vtx, vtx_size, tris, tri_size, tempRes, map, ref);

	cudaFree(tempRes);
}


__global__ void checkTri(uint3* origin, int* ptrInfo, int* triInfo, size_t vSize, size_t tSize, size_t interval) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;

	//while (tIdx < tSize) {
	if (tIdx >= tSize)
		return;

	const uint3 tri = origin[tIdx];
	triInfo[tIdx] = ((ptrInfo[tri.x] == 1) || (ptrInfo[tri.y] == 1) || (ptrInfo[tri.z] == 1)) ? 1 : 0;

	//	tIdx += interval;
	//}
}

struct is_verified {
	__host__ __device__
		bool operator()(int x) {
		return (x == 1);
	}
};

void getTriangleList(uint3* origin, int* ptrInfo, uint3* resTri, size_t vSize, size_t tSize, size_t& new_size) {
	int* triInfo;
	cudaMalloc(&triInfo, sizeof(int) * tSize);

	dim3 blockSize(ceil(tSize / 1024.), 1, 1);
	checkTri << <blockSize, 1024 >> > (origin, ptrInfo, triInfo, vSize, tSize, 1024 * 1024);
	CUDA_SYNC_CHECK();

	uint3* raw_ptr = origin;
	int* raw_ptr2 = triInfo;
	uint3* raw_ptr3 = resTri;

	thrust::device_ptr<uint3> __dataPtr(raw_ptr);
	thrust::device_ptr<int> __stancilPtr(raw_ptr2);
	thrust::device_ptr<uint3> __resultPtr(raw_ptr3);

	auto res_last = thrust::copy_if(__dataPtr, __dataPtr + tSize, __stancilPtr, __resultPtr, is_verified());
	new_size = res_last - __resultPtr;

	cudaFree(triInfo);
}

__global__ void inputNewVertices(float3* newVertices, float3* originalVertices,
	uint* deduplicationIndices, size_t* mappingIndices, size_t newVerticesSize) {

	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tIdx >= newVerticesSize)
		return;

	newVertices[tIdx] = originalVertices[deduplicationIndices[tIdx]];

	mappingIndices[deduplicationIndices[tIdx]] = tIdx;
}

__global__ void inputNewIndicesAndNormal(uint3* campactedIndices, uint3* newIndices, size_t* mappingIndices,
	float3* vertices, float3* normals, float3* v_normals, size_t indicesSize) {

	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tIdx >= indicesSize)
		return;

	auto& idx = newIndices[tIdx];
	idx = make_uint3(mappingIndices[campactedIndices[tIdx].x], mappingIndices[campactedIndices[tIdx].y], mappingIndices[campactedIndices[tIdx].z]);
	normals[tIdx] = normalize(cross(vertices[idx.y] - vertices[idx.x], vertices[idx.z] - vertices[idx.x]));

	atomicAdd(&v_normals[idx.x].x, normals[tIdx].x);
	atomicAdd(&v_normals[idx.x].y, normals[tIdx].y);
	atomicAdd(&v_normals[idx.x].z, normals[tIdx].z);

	atomicAdd(&v_normals[idx.y].x, normals[tIdx].x);
	atomicAdd(&v_normals[idx.y].y, normals[tIdx].y);
	atomicAdd(&v_normals[idx.y].z, normals[tIdx].z);

	atomicAdd(&v_normals[idx.z].x, normals[tIdx].x);
	atomicAdd(&v_normals[idx.z].y, normals[tIdx].y);
	atomicAdd(&v_normals[idx.z].z, normals[tIdx].z);
}

__global__ void normalize(float3* normalized, size_t indicesSize) {
	size_t tIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tIdx >= indicesSize)
		return;

	normalized[tIdx] = normalize(normalized[tIdx]);
}

void cudaSubdivide(float3* originVTX, uint3* originTRI, uint3* subTRI, size_t subTRISize, float3* &outputVTX, uint3* &outputTRI, size_t &outputVSize, size_t &outputTSize) {
	CUDABuffer newIndicies;
	newIndicies.alloc(sizeof(uint3) * subTRISize);

	thrust::device_ptr<uint> dPtr(&subTRI[0].x);
	thrust::device_vector<uint> dVec(dPtr, dPtr + 3 * subTRISize);
	thrust::device_vector<int> ones(3 * subTRISize, 1);

	thrust::sort(dVec.begin(), dVec.end());

	thrust::device_vector<uint> deduplicationIndices(3 * subTRISize);
	thrust::device_vector<int> counts(3 * subTRISize);

	auto reduceResult = thrust::reduce_by_key(dVec.begin(), dVec.end(), ones.begin(), deduplicationIndices.begin(), counts.begin());
	deduplicationIndices.erase(reduceResult.first, deduplicationIndices.end());

	size_t newVerticesSize = deduplicationIndices.size();
	cudaDeviceSynchronize();

	CUDABuffer newVertices;
	newVertices.alloc(sizeof(float3) * newVerticesSize);

	size_t maxIndices = *thrust::max_element(deduplicationIndices.begin(), deduplicationIndices.end());
	CUDABuffer mappingIndices;
	mappingIndices.alloc(maxIndices * sizeof(size_t));

	dim3 blockDim(1024, 1, 1);
	dim3 gridDim(std::ceil((float)newVerticesSize / (float)blockDim.x), 1, 1);
	inputNewVertices << <gridDim, blockDim >> > ((float3*)newVertices.d_pointer(), originVTX,
		(uint*)thrust::raw_pointer_cast(deduplicationIndices.data()), (size_t*)mappingIndices.d_pointer(), newVerticesSize);
	cudaStreamSynchronize(0);

	// 4 : compacted indices�� new vertices�� ���缭 new indices ����
		// 5 : new vertices�� normal �� ���ϱ�
		// 6.1 : vertex_normal add
		// 6.2 : vertex_normal count 
	CUDABuffer normalBuffer;
	normalBuffer.alloc(sizeof(float3) * subTRISize);
	CUDABuffer n_vBuffer;
	n_vBuffer.alloc(sizeof(float3) * subTRISize);
	cudaMemset((void*)n_vBuffer.d_pointer(), 0, n_vBuffer.sizeInBytes);

	gridDim.x = std::ceil((float)subTRISize / (float)blockDim.x);
	inputNewIndicesAndNormal << <gridDim, blockDim >> > (subTRI, (uint3*)newIndicies.d_pointer(),
		(size_t*)mappingIndices.d_pointer(), (float3*)newVertices.d_pointer(), (float3*)normalBuffer.d_pointer(), (float3*)n_vBuffer.d_pointer(), subTRISize);
	cudaStreamSynchronize(0);

	//// 6.2 : normalization
	normalize << <gridDim, blockDim >> > ((float3*)n_vBuffer.d_pointer(), subTRISize);
	cudaStreamSynchronize(0);

	outputVTX = (float3*)newVertices.d_pointer();
	outputTRI = (uint3*)newIndicies.d_pointer();
	outputVSize = newVerticesSize;
	outputTSize = subTRISize;

	mappingIndices.free();
	normalBuffer.free();
	n_vBuffer.free();
}