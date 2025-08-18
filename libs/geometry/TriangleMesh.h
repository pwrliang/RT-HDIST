#pragma once
#include <vector>
#include <string>
#include "3rdParty/helper_math.h"

class Material_t {
public:
    float3 ambient = { 1,1,1 };
    float3 diffuse = { 1,1,1 };
    float3 specular = { 0.5,0.5,0.5 };
    float3 emmision = { 0,0,0 };
    float shininess = 250;
    float dissolve = 1;
    int illuminationModel = 2;
};

class TriangleMesh {
public:
	std::vector<float3> vertex;
	std::vector<float3> normal;
	std::vector<float2> texcoord;
	std::vector<uint3> index;

    std::string name;

    int materialID{ -1 };
    int materialTextureID{ -1 };
};


struct Texture {
    ~Texture()
    {
        if (pixel) delete[] pixel;
    }

    uint32_t* pixel{ nullptr };
    int2     resolution{ -1, -1 };
};

struct Model {
    Model(){}
    ~Model()
    {
        for (auto mesh : meshes) delete mesh;
        for (auto material : materials) delete material;
        for (auto texture : textures) delete texture;
    }

    std::vector<Material_t*> materials;
    std::vector<TriangleMesh*> meshes;
    std::vector<Texture*>      textures;

        // 깊은 복사 생성자
    Model(const Model& other) {
        for (auto m : other.materials)
            materials.push_back(new Material_t(*m));
        for (auto mesh : other.meshes)
            meshes.push_back(new TriangleMesh(*mesh));
        for (auto tex : other.textures)
            textures.push_back(new Texture(*tex));
    }

    // 깊은 복사 대입 연산자
    Model& operator=(const Model& other) {
        if (this != &other) {
            // 기존 자원 해제
            for (auto m : materials) delete m;
            for (auto mesh : meshes) delete mesh;
            for (auto tex : textures) delete tex;
            materials.clear();
            meshes.clear();
            textures.clear();

            // 깊은 복사
            for (auto m : other.materials)
                materials.push_back(new Material_t(*m));
            for (auto mesh : other.meshes)
                meshes.push_back(new TriangleMesh(*mesh));
            for (auto tex : other.textures)
                textures.push_back(new Texture(*tex));
        }
        return *this;
    }
};