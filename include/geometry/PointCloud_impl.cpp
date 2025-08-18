#include "PointCloud.h"
#include "3rdParty/IO.h"
#include <iostream>
#include <map>
#include <unordered_map>
#include <filesystem>
#include <random>
#include "3rdParty/tinyobj_loader.h"
#include <set>

template<typename T>
inline T ByteSwap(T val) {
	uint8_t* bytes = reinterpret_cast<uint8_t*>(&val);
	std::reverse(bytes, bytes + sizeof(T));
	return val;
}

template <>
PointCloud IO::read<PointCloud,SPIN::PLY>(std::string filename) {
	std::ifstream plys(filename, std::ios::binary);

	if (!plys.is_open())
	{
		std::cerr << "File can`t open ! : " << filename << std::endl;
		return {};
	}

	PointCloud pcd;

	std::string encode_type = "ascii";
	std::string encode_version = "1.0";

	size_t vertex_size = 0;

	std::string sGet;
	while (sGet != "end_header") {
		plys >> sGet;

		if (sGet == "format") {
			plys >> encode_type;
			plys >> encode_version;
		}

		if (sGet == "element") {
			plys >> sGet;
			if (sGet == "vertex") {
				plys >> vertex_size;
			}
		}
	}

	pcd.points.resize(vertex_size);

	float3 _min = { 1e8f, 1e8f, 1e8f }, _max = { -1e8f, -1e8f, -1e8f };

	for (int i = 0; i < vertex_size; i++) {
		if (encode_type == "ascii") {
			float3* pts = &pcd.points[i];
			plys >> pts->x;
			plys >> pts->y;
			plys >> pts->z;
			_min = fminf(_min, *pts);
			_max = fmaxf(_max, *pts);
		}
		else if (encode_type == "binary_little_endian") {
			float3& pts = pcd.points[i];
			plys.read(reinterpret_cast<char*>(&pts.x), sizeof(float));
			plys.read(reinterpret_cast<char*>(&pts.y), sizeof(float));
			plys.read(reinterpret_cast<char*>(&pts.z), sizeof(float));

			if (!plys.good()) {
				std::cerr << "Error reading binary data for vertex " << i << std::endl;
				break;
			}

			pts.x = ByteSwap(pts.x);
			pts.y = ByteSwap(pts.y);
			pts.z = ByteSwap(pts.z);

			//std::cout << pts.x << " " << pts.y << " " << pts.z << std::endl;

			_min = fminf(_min, pts);
			_max = fmaxf(_max, pts);
		}
	}

	pcd.bounds = { _min, _max };


	plys.close();

	return pcd;
}

template <>
PointCloud IO::read<PointCloud, SPIN::OBJ>(std::string filename) {
	PointCloud pcd;

	std::filesystem::path path = filename;
	const std::string modelDir = path.parent_path().string() + "/";

	tinyobj::attrib_t attributes;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err = "";

	std::cout << modelDir << std::endl;

	bool readOK
		= tinyobj::LoadObj(&attributes,
			&shapes,
			&materials,
			&err,
			&err,
			filename.c_str(),
			modelDir.c_str(),
			/* triangulate */true);
	if (!readOK) {
		throw std::runtime_error("Could not read OBJ model from " + filename + " : " + err);
	}

	if (materials.empty())
		throw std::runtime_error("could not parse materials ...");


	std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;

	float3 _min = { 1e8f, 1e8f, 1e8f }, _max = { -1e8f, -1e8f, -1e8f };

	for (int i = 0; i < attributes.vertices.size() / 3; i++) {
		pcd.points.push_back({ attributes.vertices[i * 3 + 0],attributes.vertices[i * 3 + 1] ,attributes.vertices[i * 3 + 2] });

		_min = fminf(_min, pcd.points[i]);
		_max = fmaxf(_max, pcd.points[i]);
	}
	pcd.bounds = { _min, _max };
	
	return pcd;
}
template <>
int IO::write<PointCloud,SPIN::PLY>(std::string filename, PointCloud& data) {
	std::ofstream plys(filename);

	if (!plys.is_open())
		return -1;

	plys << "ply" << std::endl;
	plys << "format ascii 1.0" << std::endl;
	plys << "element vertex " << data.points.size() << std::endl;
	plys << "property float x" << std::endl;
	plys << "property float y" << std::endl;
	plys << "property float z" << std::endl;
	plys << "end_header" << std::endl;

	for (auto& v : data.points) {
		plys << v.x << " " << v.y << " " << v.z << std::endl;
	}

	plys.close();

	return 0;
}

PointCloud::PointCloud(const std::string& filePath) {
	using Path = std::filesystem::path;

	PointCloud reads;

	Path path(filePath);
	if (path.extension() == ".ply") {
		reads = IO::read<PointCloud, SPIN::PLY>(filePath);
	}
	else if (path.extension() == ".obj") {
		reads = IO::read<PointCloud, SPIN::OBJ>(filePath);
	}

	*this = reads;
}

int PointCloud::write(std::string& filePath) {
	using Path = std::filesystem::path;

	int res = 0;

	Path path(filePath);
	if (path.extension() == ".ply") {
		res = IO::write<PointCloud, SPIN::PLY>(filePath, *this);
	}

	return res;
}

void PointCloud::translatef(float3 move) {
	bounds._min += move;
	bounds._max += move;

	for (auto& v : points) {
		v += move;
	}
}

PointCloud genRandom(float3 min, float3 max, size_t count, std::mt19937 &random_machine){
	PointCloud pd;

	pd.bounds._min = min;
	pd.bounds._max = max;
	pd.points.resize(count);
	for (int i = 0; i < count; i++) {
		float3 pts;
		for (int j = 0; j < 3; j++) {
			float* __minx = &min.x;
			float* __maxx = &max.x;
			std::uniform_real_distribution<> dis(__minx[j], __maxx[j]);
			
			float* __ptsx = &pts.x;
			__ptsx[j] = dis(random_machine);
		}
		pd.points[i] = pts;
	}
	
	return pd;
}
