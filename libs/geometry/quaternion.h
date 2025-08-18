#pragma once
#include "3rdParty/helper_math.h"
#include <memory>
typedef float4 quat;

void quat2rom(quat q, float* rom);

#ifdef QUATERNION_IMPLEMENTATION
void quat2rom(quat q, float* rom) {
	float3 rotMat[3];

	rotMat[0] = make_float3(1 - 2 * (q.y * q.y + q.z * q.z),
		2 * (q.x * q.y - q.w * q.z),
		2 * (q.x * q.z + q.w * q.y));
	rotMat[1] = make_float3(2 * (q.x * q.y + q.w * q.z),
		1 - 2 * (q.x * q.x + q.z * q.z),
		2 * (q.y * q.z - q.w * q.x));
	rotMat[2] = make_float3(2 * (q.x * q.z - q.w * q.y),
		2 * (q.y * q.z + q.w * q.x),
		1 - 2 * (q.x * q.x + q.y * q.y));

	float _rom[9] = {
		rotMat[0].x, rotMat[1].x, rotMat[2].x,
		rotMat[0].y, rotMat[1].y, rotMat[2].y,
		rotMat[0].z, rotMat[1].z, rotMat[2].z
	};

	memcpy(rom, _rom, sizeof(float) * 9);
}
#endif