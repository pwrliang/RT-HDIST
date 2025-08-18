#pragma once
#include <string>
#include <filesystem>
#include <fstream>

namespace SPIN {
	class FileTypes {
	public:
		std::string extension = "";
	};

	class PCD : public FileTypes {
	public:
		PCD() {
			extension = ".pcd";
		}
	};

	class PLY : public FileTypes {
	public:
		PLY() {
			extension = ".ply";
		}
	};

	class OBJ : public FileTypes {
	public:
		OBJ() {
			extension = ".obj";
		}
	};
}

class IO {
	typedef int ErrorCode;
public:
	template <typename T, typename FileType>
	inline static T read(std::string filename);

	template <typename T, typename FileType>
	inline static ErrorCode write(std::string fileName, T& data);
};
