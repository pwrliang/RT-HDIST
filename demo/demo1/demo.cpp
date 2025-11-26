#include <iostream>
#include "3rdParty/helper_math.h"
#include "OptiX_Base.h"
#include "rtHausdorffQCluster.h"
#include "img_loader.h"
#include "ply_loader.h"

#include "3rdParty/TimeChecker.h"
#include "3rdParty/Logger.h"
#include "3rdParty/IO.h"

#define GLOBALPARAM_IMPLEMENTATION
#include "GlobalParam.h"

#include "PointCloud.h"
#include "Object_t.h"

#include "MortonUtils.h"
#include "ReduceUtils.h"
#include "AABBSupport.h"

std::mt19937 random_machine;
std::map<std::string, int> globalParams;

std::string loggerPath;
float3 globalTransform;
float globalTransformRatio;

std::vector<std::string> inputFilePaths;

void buildQClusterShader();

float3 randomDir() {
	float3 dir;

	std::normal_distribution<float> nd(0.0f, 1.0f);
	dir.x = nd(random_machine);
	dir.y = nd(random_machine);
	dir.z = nd(random_machine);

	dir /= length(dir);

	return dir;
}
bool endsWith(const std::string& fullString, const std::string& ending) {
	// 1. Check if the ending is longer than the full string.
	if (ending.size() > fullString.size()) {
		return false;
	}

	// 2. Compare the ending part of fullString with ending.
	// std::string::compare(pos, len, str) compares the substring of
	// fullString starting at pos (and with length len) with str.
	// The comparison starts at: fullString.size() - ending.size()
	// The length of the substring to compare is: ending.size()
	return fullString.compare(
		fullString.size() - ending.size(), // Start position
		ending.size(),                     // Length to compare
		ending                             // String to compare against
	) == 0; // compare returns 0 if the strings are equal
}
HDGPUParam<HDMODE::POINT> ReadPoints(const std::string& inputFilePath,
	float3 boxTranslate = {0,0,0}) {
	HDGPUParam<HDMODE::POINT> points;
	if (endsWith(inputFilePath, ".obj")) {
		Object_t hA = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[0]);
		for (auto &v: hA.model->meshes[0]->vertex) {
			v += boxTranslate;
		}
		cudaMalloc(&points.vert, sizeof(float3) * hA.model->meshes[0]->vertex.size());
		upload(hA.model->meshes[0]->vertex, points);
		points.vSize = hA.model->meshes[0]->vertex.size();
	} else if (endsWith(inputFilePath, ".ply")) {
		auto h_points = LoadPLY(inputFilePath);
		for (auto &p:h_points) {
			p += boxTranslate;
		}
		cudaMalloc(&points.vert, sizeof(float3) * h_points.size());
		upload(h_points, points);
		points.vSize = h_points.size();
	} else if (endsWith(inputFilePath, ".nii")) {
		itk::Size<3> img_size;
		auto h_points = LoadImage(inputFilePath, img_size);
		for (auto &p:h_points) {
			p += boxTranslate;
		}
		cudaMalloc(&points.vert, sizeof(float3) * h_points.size());
		upload(h_points, points);
		points.vSize = h_points.size();
	} else {
		throw std::invalid_argument("Unsupported input file type");
	}
	return points;
}


int main(int argc, char* argv[])
{
	std::cout << "Device name : " << optixGlobalParams.deviceProps.name << std::endl;
    buildQClusterShader();

	verifyArguments(argc, argv);

    std::random_device rd;
	unsigned int seed = (globalParams["seed"] >= 0) ? globalParams["seed"]:rd();
	std::cout << "Random seed : " << seed << std::endl;
	random_machine = std::mt19937(seed);


    SPIN::Logger log;



	HDGPUParam<HDMODE::POINT> dA = ReadPoints(inputFilePaths[0]);
	HDGPUParam<HDMODE::POINT> dB;

	float3 boxTranslate = { 0,0,0 };
	OptixAabb aabb = computeAABB_device(dA.vert, dA.vSize);
	float3 aabbSize = aabb2size(aabb);

	if (globalParams["obj_count"] == 1) {
		if (globalParams["translate_ratio"]) {
			globalTransform = globalTransformRatio * make_float3(aabbSize.x, 0, 0);
			boxTranslate = globalTransform;
		}
		dB = ReadPoints(inputFilePaths[0], boxTranslate);
	} else {
		dB = ReadPoints(inputFilePaths[1]);
	}

	log.data["00_source_count"].push_back((int)dA.vSize);
	log.data["00_target_count"].push_back((int)dB.vSize);

	float3 Amin = aabb2min(aabb);
	float3 Amax = aabb2min(aabb);

	log.data["01_source_aabb_min_x"].push_back((float)Amin.x);
	log.data["01_source_aabb_min_y"].push_back((float)Amin.y);
	log.data["01_source_aabb_min_z"].push_back((float)Amin.z);

	log.data["01_source_aabb_max_x"].push_back((float)Amax.x);
	log.data["01_source_aabb_max_y"].push_back((float)Amax.y);
	log.data["01_source_aabb_max_z"].push_back((float)Amax.z);

	aabb = computeAABB_device(dB.vert, dB.vSize);
	float3 Bmin = aabb2min(aabb);
	float3 Bmax = aabb2min(aabb);

	log.data["01_target_aabb_min_x"].push_back((float)Bmin.x);
	log.data["01_target_aabb_min_y"].push_back((float)Bmin.y);
	log.data["01_target_aabb_min_z"].push_back((float)Bmin.z);

	log.data["01_target_aabb_max_x"].push_back((float)Bmax.x);
	log.data["01_target_aabb_max_y"].push_back((float)Bmax.y);
	log.data["01_target_aabb_max_z"].push_back((float)Bmax.z);

	log.data["02_Bit_count_AtoB"].push_back(globalParams["grid_1"]);
	log.data["02_Bit_count_BtoA"].push_back(globalParams["grid_2"]);

    float HD;
	float3 cand1, cand2;
	std::map<std::string, float> timeParam;

	timeParam["IndexSpaceBuildTime"] = 0;
	timeParam["FilteringTime"] = 0;
	timeParam["ComputingTime"] = 0;

	auto RTQclusterTimes = SPIN::TimeCheck([&]() {
		float3 tempCand1, tempCand2;
		float HD1 = qclusterHD(static_cast<OptiXHDProgram&>(*optixGlobalParams.programList["QCluster"]), dA, dB, cand1, cand2, sqrt(3), globalParams["grid_1"], timeParam);
		// float HD2 = qclusterHD(static_cast<OptiXHDProgram&>(*optixGlobalParams.programList["QCluster"]), dB, dA, tempCand1, tempCand2, sqrt(3), globalParams["grid_2"], timeParam);
		// std::cout << HD1 << ", " << HD2 << std::endl;

		// if (HD2 > HD1) {
		// 	cand1 = tempCand1;
		// 	cand2 = tempCand2;
		// }

		// HD = fmaxf(HD1, HD2);
		std::cout << "HD1 " << HD1 << std::endl;
		HD = HD1;
    });

	log.data["04_HD"].push_back(HD);
	log.data["05_Performance(ms)"].push_back(RTQclusterTimes);

	log.data["05__01_index_time"].push_back(timeParam["IndexSpaceBuildTime"]);
	log.data["05__02_filtering_time"].push_back(timeParam["FilteringTime"]);
	log.data["05__03_computing_time"].push_back(timeParam["ComputingTime"]);

	log.data["08_cand1_x"].push_back(cand1.x);
	log.data["08_cand1_y"].push_back(cand1.y);
	log.data["08_cand1_z"].push_back(cand1.z);

	log.data["08_cand2_x"].push_back(cand2.x);
	log.data["08_cand2_y"].push_back(cand2.y);
	log.data["08_cand2_z"].push_back(cand2.z);

	std::cout << cand1.x << " " << cand1.y << " " << cand1.z << std::endl;
	std::cout << cand2.x << " " << cand2.y << " " << cand2.z << std::endl;

	//Logger write---------------------------------------------------------------------------------------------------------------------------------------//
	bool fileExists = std::filesystem::exists(loggerPath);

	std::ofstream logOut(loggerPath, std::ios::app);

	if (!fileExists) {
		logOut << log;
	}
	else {
		int t_size = log.data.begin()->second.size();
		for (int i = 0; i < t_size; i++) {
			for (auto& v : log.data) {
				std::visit([&logOut](auto&& arg) {logOut << arg << ";"; }, v.second[i]);
			}
			logOut << std::endl;
		}
	}
	logOut.close();

	return 0;
}

void buildQClusterShader(){
	OptiXProgramCompileOption hdShaderOption;
	hdShaderOption.fileName = "__shader__hd__qcluster__";
	hdShaderOption.filePath = "";
	hdShaderOption.rayCount = 1;
	hdShaderOption.launchParamName = "optixLaunchParams";
	hdShaderOption.rayGenName = "__raygen__program__";
	hdShaderOption.missProgramNames = { "__miss__radiance" };
	hdShaderOption.hitProgramCount = 1;
	hdShaderOption.hitProgramNames = { {"__intersection__radiance", "__anyhit__radiance", "__closesthit__radiance"} };

	OptiXHDProgram* HDProgram = new OptiXHDProgram(hdShaderOption);

	optixGlobalParams.programList["QCluster"] = HDProgram;
}