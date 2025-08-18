#pragma once
#include "3rdParty/helper_math.h"

#include "quaternion.h"
#include "TriangleMesh.h"

class Transform_t {
public:
	float3 position = {0,0,0};
	quat rotation = {0,0,0,1};
	float3 scale = {1,1,1};

	void getTRS(float* trs);
};

class Object_t{
public:
	Model *model;

	Object_t(){};
	Object_t(const std::string& filename);
	~Object_t() { delete model; }
};

class Instance_t : public Transform_t {
public:
	Object_t* obj;

	Instance_t* parent;
};

class Scene {
public:
	std::vector<Instance_t*> hierarchy;
};
