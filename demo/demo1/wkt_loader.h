#pragma once


#include <dirent.h>
#include <sys/stat.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/box.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <vector>


std::vector<float3> LoadWKTPoints(
    const std::string &path, int limit = std::numeric_limits<int>::max()) {
    std::ifstream ifs(path);
    std::string line;
    using boost_point_t =
            boost::geometry::model::point<float, 2,
                boost::geometry::cs::cartesian>;
    using boost_polygon_t = boost::geometry::model::polygon<boost_point_t>;
    using boost_linestring_t = boost::geometry::model::linestring<boost_point_t>;
    using point_t = float3;
    std::vector<point_t> points;

    while (std::getline(ifs, line)) {
        if (!line.empty()) {
            if (line.rfind("MULTIPOLYGON", 0) == 0) {
                boost::geometry::model::multi_polygon<boost_polygon_t> multi_poly;
                boost::geometry::read_wkt(line, multi_poly);

                for (auto &poly: multi_poly) {
                    for (auto &p: poly.outer()) {
                        point_t fp{p.get<0>(), p.get<1>(), 0};
                        points.push_back(fp);
                        if (points.size() >= limit) {
                            break;
                        }
                    }
                }
            } else if (line.rfind("POLYGON", 0) == 0) {
                boost_polygon_t poly;
                boost::geometry::read_wkt(line, poly);

                for (auto &p: poly.outer()) {
                    point_t fp{p.get<0>(), p.get<1>(), 0};
                    points.push_back(fp);
                    if (points.size() >= limit) {
                        break;
                    }
                }
            } else if (line.rfind("LINESTRING") == 0) {
                boost_linestring_t line_string;
                boost::geometry::read_wkt(line, line_string);
                for (auto &p: line_string) {
                    point_t fp{p.get<0>(), p.get<1>(), 0};
                    points.push_back(fp);
                    if (points.size() >= limit) {
                        break;
                    }
                }
            } else if (line.rfind("MULTILINESTRING") == 0) {
                boost::geometry::model::multi_linestring<boost_linestring_t> multi_l;
                boost::geometry::read_wkt(line, multi_l);


                for (auto &l: multi_l) {
                    for (auto &p: l) {
                        point_t fp{p.get<0>(), p.get<1>(), 0};

                        points.push_back(fp);
                        if (points.size() >= limit) {
                            break;
                        }
                    }
                }
            } else if (line.rfind("POINT", 0) == 0) {
                boost_point_t p;
                boost::geometry::read_wkt(line, p);
                point_t fp{p.get<0>(), p.get<1>(), 0};

                points.push_back(fp);
                if (points.size() >= limit) {
                    break;
                }
            } else if (line.rfind("GEOMETRYCOLLECTION", 0) == 0) {
                // TODO
            } else {
                std::cerr << "Bad Geometry " << line << "\n";
                abort();
            }
        }
    }
    ifs.close();
    return points;
}

template<typename POINT_T>
void SerializePoints(const char *file, const std::vector<POINT_T> &points) {
    std::ofstream ofs(file, std::ios::binary);

    size_t size = points.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));

    // Write the vector data
    if (size > 0) {
        ofs.write(reinterpret_cast<const char *>(points.data()),
                  sizeof(POINT_T) * size);
    }

    ofs.close();
}

template<typename POINT_T>
std::vector<POINT_T> DeserializePoints(const char *file) {
    std::vector<POINT_T> deserialized_points;
    std::ifstream ifs(file, std::ios::binary);
    // Read the size first
    size_t size = 0;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Resize the vector to hold the data
    deserialized_points.resize(size);

    // Read the vector data
    if (size > 0) {
        ifs.read(reinterpret_cast<char *>(deserialized_points.data()),
                 sizeof(POINT_T) * size);
    }

    ifs.close();
    return deserialized_points;
}

std::vector<float3> LoadPoints(
    const std::string &path, const std::string &serialize_prefix,
    int limit = std::numeric_limits<int>::max()) {
    using point_t = float3;
    std::string escaped_path;
    std::replace_copy(path.begin(), path.end(), std::back_inserter(escaped_path),
                      '/', '_');

    if (!serialize_prefix.empty()) {
        DIR *dir = opendir(serialize_prefix.c_str());
        if (dir) {
            closedir(dir);
        } else if (ENOENT == errno) {
            if (mkdir(serialize_prefix.c_str(), 0755)) {
                std::cerr << "Cannot create dir " << path;
                abort();
            }
        } else {
            std::cerr << "Cannot open dir " << path;
            abort();
        }
    }
    auto ser_path = serialize_prefix + "/points_" + escaped_path + "_limit_" +
                    std::to_string(limit) + ".bin";

    std::vector<point_t> points;

    if (access(ser_path.c_str(), R_OK) == 0) {
        points = DeserializePoints<point_t>(ser_path.c_str());
    } else {
        points = LoadWKTPoints(path, limit);

        if (!points.empty() && !serialize_prefix.empty()) {
            SerializePoints<point_t>(ser_path.c_str(), points);
        }
    }
    return points;
}
