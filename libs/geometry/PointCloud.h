#pragma once
#include "3rdParty/helper_math.h"
#include <vector>
#include <string>
#include <random>

class PointCloud {
	typedef struct Bound {
		float3 _min;
		float3 _max;
	};
public:
	std::vector<float3> points;
	Bound bounds = {};

	PointCloud() {};
	PointCloud(const std::string& filePath);

	int write(std::string& filePath);

	void translatef(float3 move);
};

PointCloud genRandom(float3 min, float3 max, size_t count, std::mt19937 &random_machine);