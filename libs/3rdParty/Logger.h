#pragma once
#include <filesystem>
#include <map>
#include <any>
#include <variant>

namespace SPIN {

	class Logger {
	public:
		std::map<std::string, std::vector<std::variant<int, float, double, std::string>>> data;
		std::string filePath;

		void print();
		friend std::ostream& operator<<(std::ostream& os, const Logger& data);

	};

	std::ostream& operator<<(std::ostream& os, const Logger& data) {
		int t_size = data.data.begin()->second.size(); // it assumes that all of the map vector sizes is same 

		os << "sep=;" << std::endl;
		for (auto& v : data.data) {
			os << v.first << ";";
		}
		os << std::endl;

		for (int i = 0; i < t_size; i++) {
			for (auto& v : data.data) {
				std::visit([&os](auto&& arg) {os << arg << ";"; }, v.second[i]);
			}
			os << std::endl;
		}

		return os;
	}

}