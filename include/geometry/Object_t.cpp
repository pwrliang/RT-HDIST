#define QUATERNION_IMPLEMENTATION
#include "Object_t.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "3rdParty/tinyobj_loader.h"
#include "3rdParty/helper_math.h"

#define STB_IMAGE_IMPLEMENTATION
#include "3rdParty/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "3rdParty/stb_image_write.h"

#include <algorithm>
#include <set>
#include <iostream>
#include <filesystem>

#include "3rdParty/IO.h"


std::vector<float3> basicCUBE = {
    {1,1,-1},
    {1,-1,-1},
    {1,1,1},
    {1,-1,1},
    {-1,1,-1},
    {-1,-1,-1},
    {-1,1,1},
    {-1,-1,1}
};

std::vector<uint3> CUBEIDX = {
    {5,3,1},	{3,8,4},
    {7,6,8},    {2,8,6},
    {1,4,2},    {5,2,6},
    {5,7,3},    {3,7,8},
    {7,5,6},    {2,4,8},
    {1,3,4},    {5,1,2}
};


template<typename T>
inline T ByteSwap(T val) {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&val);
    std::reverse(bytes, bytes + sizeof(T));
    return val;
}

namespace std {
    inline bool operator<(const tinyobj::index_t& a,
        const tinyobj::index_t& b)
    {
        if (a.vertex_index < b.vertex_index) return true;
        if (a.vertex_index > b.vertex_index) return false;

        if (a.normal_index < b.normal_index) return true;
        if (a.normal_index > b.normal_index) return false;

        if (a.texcoord_index < b.texcoord_index) return true;
        if (a.texcoord_index > b.texcoord_index) return false;

        return false;
    }
}

/*! find vertex with given position, normal, texcoord, and return
    its vertex ID, or, if it doesn't exit, add it to the mesh, and
    its just-created index */
int addVertex(TriangleMesh* mesh,
    tinyobj::attrib_t& attributes,
    const tinyobj::index_t& idx,
    std::map<int, int>& knownVertices)
{
    if (knownVertices.find(idx.vertex_index) != knownVertices.end())
        return knownVertices[idx.vertex_index];

    const float3* vertex_array = (const float3*)attributes.vertices.data();
    const float3* normal_array = (const float3*)attributes.normals.data();
    const float2* texcoord_array = (const float2*)attributes.texcoords.data();

    int newID = (int)mesh->vertex.size();
    knownVertices[idx.vertex_index] = newID;

    mesh->vertex.push_back(vertex_array[idx.vertex_index]);
    if (idx.normal_index >= 0) {
        while (mesh->normal.size() < mesh->vertex.size())
            mesh->normal.push_back(normal_array[idx.normal_index]);
    }
    if (idx.texcoord_index >= 0) {
        while (mesh->texcoord.size() < mesh->vertex.size())
            mesh->texcoord.push_back(texcoord_array[idx.texcoord_index]);
    }

    // just for sanity's sake:
    if (mesh->texcoord.size() > 0)
        mesh->texcoord.resize(mesh->vertex.size());
    // just for sanity's sake:
    if (mesh->normal.size() > 0)
        mesh->normal.resize(mesh->vertex.size());

    return newID;
}

/*! load a texture (if not already loaded), and return its ID in the
    model's textures[] vector. Textures that could not get loaded
    return -1 */
int loadTexture(Model* model,
    std::map<std::string, int>& knownTextures,
    const std::string& inFileName,
    const std::string& modelPath)
{
    if (inFileName == "")
        return -1;

    if (knownTextures.find(inFileName) != knownTextures.end())
        return knownTextures[inFileName];

    std::string fileName = inFileName;
    // first, fix backspaces:
    for (auto& c : fileName)
        if (c == '\\') c = '/';
    fileName = modelPath + "/" + fileName;

    int2 res;
    int   comp;
    unsigned char* image = stbi_load(fileName.c_str(),
        &res.x, &res.y, &comp, STBI_rgb_alpha);
    int textureID = -1;
    if (image) {
        textureID = (int)model->textures.size();
        Texture* texture = new Texture;
        texture->resolution = res;
        texture->pixel = (uint32_t*)image;

        /* iw - actually, it seems that stbi loads the pictures
           mirrored along the y axis - mirror them here */
        for (int y = 0; y < res.y / 2; y++) {
            uint32_t* line_y = texture->pixel + y * res.x;
            uint32_t* mirrored_y = texture->pixel + (res.y - 1 - y) * res.x;
            int mirror_y = res.y - 1 - y;
            for (int x = 0; x < res.x; x++) {
                std::swap(line_y[x], mirrored_y[x]);
            }
        }

        model->textures.push_back(texture);
    }
    else {
        std::cout << "Could not load texture from " << modelPath + "/" + fileName << "!" << std::endl;
    }

    knownTextures[inFileName] = textureID;
    return textureID;
}

Model* loadOBJ(const std::string& objFile)
{
    Model* model = new Model;

    std::filesystem::path path(objFile);

    const std::string modelDir
        = path.parent_path().string() + "/";

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
            objFile.c_str(),
            modelDir.c_str(),
            /* triangulate */true);
    if (!readOK) {
        throw std::runtime_error("Could not read OBJ model from " + objFile + " : " + err);
    }

    if (materials.empty())
        throw std::runtime_error("could not parse materials ...");
   
    std::cout << "Done loading obj file - found " << shapes.size() << " shapes with " << materials.size() << " materials" << std::endl;
    std::map<std::string, int>      knownTextures;
    for (int shapeID = 0; shapeID < (int)shapes.size(); shapeID++) {
        tinyobj::shape_t& shape = shapes[shapeID];

        std::set<int> materialIDs;
        for (auto faceMatID : shape.mesh.material_ids)
            materialIDs.insert(faceMatID);

        std::map<int, int> knownVertices;

        for (int materialID : materialIDs) {
            TriangleMesh* mesh = new TriangleMesh;

            for (int faceID = 0; faceID < shape.mesh.material_ids.size(); faceID++) {
                if (shape.mesh.material_ids[faceID] != materialID) continue;
                tinyobj::index_t idx0 = shape.mesh.indices[3 * faceID + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[3 * faceID + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[3 * faceID + 2];

                uint3 idx = make_uint3(addVertex(mesh, attributes, idx0, knownVertices),
                    addVertex(mesh, attributes, idx1, knownVertices),
                    addVertex(mesh, attributes, idx2, knownVertices));
                mesh->index.push_back(idx);
            }

            Material_t *material = new Material_t;

            if (materials[materialID].name.length() >0 && materials[materialID].name.length() < 1000) {
                if (materials[materialID].diffuse)
                    material->diffuse = (const float3&)materials[materialID].diffuse;
                if (materials[materialID].ambient)
                    material->ambient = (const float3&)materials[materialID].ambient;
                if (materials[materialID].dissolve)
                    material->dissolve = (const float)materials[materialID].dissolve;
                if (materials[materialID].emission)
                    material->emmision = (const float3&)materials[materialID].emission;
                if (materials[materialID].illum)
                    material->illuminationModel = (const int)materials[materialID].illum;
                if (materials[materialID].shininess)
                    material->shininess = (const float)materials[materialID].shininess;
                if (materials[materialID].specular)
                    material->specular = (const float3&)materials[materialID].specular;

                if (materials[materialID].diffuse_texname.length() > 0 && materials[materialID].diffuse_texname.length() < 1000)
                mesh->materialTextureID = loadTexture(model,
                    knownTextures,
                    materials[materialID].diffuse_texname,
                    modelDir);

                mesh->materialID = materialID;
                model->materials.push_back(material);
            }
            else {
                mesh->materialID = -1;
                mesh->materialTextureID = -1;
            }

            mesh->name = shape.name;

            if (mesh->vertex.empty())
                delete mesh;
            else
                model->meshes.push_back(mesh);
        }
    }

    /* for (auto mesh : model->meshes)
         for (auto vtx : mesh->vertex)
             model->bounds.extend(vtx);*/

    std::cout << "created a total of " << model->meshes.size() << " meshes" << std::endl;
    return model;
}

//only obj file supported
Object_t::Object_t(const std::string& filename){
    model = loadOBJ(filename);
}

template<>
Object_t IO::read<Object_t, SPIN::OBJ>(std::string filename) {
    Object_t obj;

    std::string path = filename;
    obj.model = loadOBJ(path);

    return obj;
}


template <>
TriangleMesh IO::read<TriangleMesh, SPIN::PLY>(std::string filename) {
    std::ifstream plys(filename, std::ios::binary);

    if (!plys.is_open())
    {
        std::cerr << "File can`t open ! : " << filename << std::endl;
        return {};
    }

    std::filesystem::path path(filename);

    TriangleMesh mesh;

    mesh.name = path.stem().string();

    std::string encode_type = "ascii";
    std::string encode_version = "1.0";

    size_t vertex_size = 0;
    size_t index_size = 0;

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
            if (sGet == "face") {   
                plys >> index_size;
            }
        }
    }

    mesh.vertex.resize(vertex_size);
    mesh.index.resize(index_size);

    float3 _min = { 1e8f, 1e8f, 1e8f }, _max = { -1e8f, -1e8f, -1e8f };

    for (int i = 0; i < vertex_size; i++) {
        if (encode_type == "ascii") {
            float3* pts = &mesh.vertex[i];
            plys >> pts->x;
            plys >> pts->y;
            plys >> pts->z;
            _min = fminf(_min, *pts);
            _max = fmaxf(_max, *pts);
        }
        else if (encode_type == "binary_little_endian") {
            float3& pts = mesh.vertex[i];
            plys.read(reinterpret_cast<char*>(&pts.x), sizeof(float));
            plys.read(reinterpret_cast<char*>(&pts.y), sizeof(float));
            plys.read(reinterpret_cast<char*>(&pts.z), sizeof(float));

            // �б� ���� üũ
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

    for (int i = 0;i < index_size; i++) {
        if (encode_type == "ascii") {
            unsigned char count = 0;
            plys >> count;

            if (count >= 4)
            {
                std::cerr << "Currently supported only triangleMesh!!" << std::endl;
                break;
            }

            uint3* idx = &mesh.index[i];

            plys >> idx->x;
            plys >> idx->y;
            plys >> idx->z;
        }
        else if (encode_type == "binary_little_endian") {
            uint8_t count = 0;
            uint3& idx = mesh.index[i];
            plys.read(reinterpret_cast<char*>(&count), sizeof(uint8_t));

            if(count >= 64)
                plys.read(reinterpret_cast<char*>(&count), sizeof(uint8_t));

            if (count >= 4)
            {
                std::cerr << "Currently supported only triangleMesh!!" << std::endl;
                break;
            }

            plys.read(reinterpret_cast<char*>(&idx.x), sizeof(unsigned int));
            plys.read(reinterpret_cast<char*>(&idx.y), sizeof(unsigned int));
            plys.read(reinterpret_cast<char*>(&idx.z), sizeof(unsigned int));

            //std::cout << idx.x << " " << idx.y << " " << idx.z << std::endl;

            //idx = idx + 1;

            // �б� ���� üũ
            if (!plys.good()) {
                std::cerr << "Error reading binary data for vertex " << i << std::endl;
                break;
            }
        }
    }

    mesh.materialID = -1;
    mesh.materialTextureID = -1;

    plys.close();

    return mesh;
}

void Transform_t::getTRS(float* trs){
    float rotMat[9];
    quat2rom(rotation, rotMat);

    // Apply scaling to rotation matrix
    rotMat[0] *= scale.x; rotMat[1] *= scale.y; rotMat[2] *= scale.z;
    rotMat[3] *= scale.x; rotMat[4] *= scale.y; rotMat[5] *= scale.z;
    rotMat[6] *= scale.x; rotMat[7] *= scale.y; rotMat[8] *= scale.z;

    float transforms[12] = {
        rotMat[0], rotMat[1], rotMat[2], position.x,
        rotMat[3], rotMat[4], rotMat[5], position.y,
        rotMat[6], rotMat[7], rotMat[8], position.z
    };

    memcpy(trs, transforms, sizeof(float) * 12);
}
