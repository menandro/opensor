#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace sor {
	class FileReader {
	public:
		FileReader();
		~FileReader() {};

		std::vector<float> vertexArray; //3-vertex 2-texture
		std::vector<unsigned int> indexArray;
		std::vector<float> normalArray; //container only if data is available
		std::string objectName;//only if available

							   //OBJ PARSED DATA
		std::vector<float> obj_vertx, obj_verty, obj_vertz;
		std::vector<float> obj_u, obj_v;
		std::vector<float> obj_normx, obj_normy, obj_normz;
		std::vector<unsigned int> obj_tri_vertices;
		std::vector<unsigned int> obj_tri_uvs;
		std::vector<unsigned int> obj_tri_normals;

		std::vector<unsigned int> obj_face_vertices;
		std::vector<unsigned int> obj_face_uvs;
		std::vector<unsigned int> obj_face_normals;

		float scale;

		bool fileExists(const std::string& filename);
		int readObj(std::string filename, float scale = 0.01f);
		int splitLines(std::fstream objectFile);
		int setScale(float scale);

		std::vector<std::string> lines;
	};
};
