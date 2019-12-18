#ifndef _CPP_IMPORTER_
#define _CPP_IMPORTER_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>

#include <fcntl.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

using std::ofstream;
using std::ifstream;

void splitString(const std::string& s, const std::string& c, std::vector<std::string>& v)
{
    std::string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));
        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

/**
 * 用 memcpy 读取文件
 */
char* loadFile(const std::string& filePath) {
    size_t MEMCPY_SIZE = 1024 * 1024;
    int fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1)
    {
        std::cerr << "Failed open()" << std::endl;
    }
    off_t file_length = lseek(fd, 0, SEEK_END); // 读取文件大小
    if (file_length == (off_t)-1)
    {
        std::cerr << "Failed lseek()" << std::endl;
    }
    void *map = mmap(NULL, file_length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED)
    {
        std::cerr << "Failed mmap()" << std::endl;
    }
    const uint8_t *file_map_start = static_cast<const uint8_t *>(map);

    char *buffer = new char[file_length];
    memcpy(buffer, file_map_start, file_length);
    return buffer;
}

void loadFeatures(const std::string &pathFeatures, std::map<std::string, int> &features, int &maxFeaIdx)
{
    ifstream in(pathFeatures.c_str());
    std::string line;
    if (in.is_open())
    {
        while (in.good())
        {
            getline(in, line);
            if (line.empty())
                continue;
            std::vector<std::string> fields;
            splitString(line, "\t", fields);
            if (fields.size() < 2)
                continue;
            std::string feaVal = fields[0];
            int feaIdx = atoi(fields[1].c_str());
            features.insert(std::make_pair(feaVal, feaIdx));
            maxFeaIdx = (feaIdx > maxFeaIdx) ? feaIdx : maxFeaIdx;
        }
    }
}

void loadInstances(std::string pathInstances, std::vector<std::vector<int> > &X, std::vector<double> &y)
{
    ifstream in(pathInstances.c_str());
    std::string line;
    if (in.is_open())
    {
        while (in.good())
        {
            getline(in, line);
            if (line.empty())
                continue;
            std::vector<std::string> fields;
            splitString(line, "\t", fields);
            if (fields.size() < 2)
                continue;
            double yi = atof(fields[0].c_str());
            y.push_back(yi);
            std::vector<int> xi;
            std::vector<std::string> xiFields;
            splitString(fields[1], ",", xiFields);
            for (int i = 0; i < xiFields.size(); ++i)
            {
                xi.push_back(atoi(xiFields[i].c_str()));
            }
            X.push_back(xi);
        }
    }
}

void outputModel(std::string pathModel, std::vector<double> &w, std::map<std::string, int> &features)
{
    ofstream out(pathModel.c_str());
    for (std::map<std::string, int>::iterator it = features.begin(); it != features.end(); ++it)
    {
        out << it->first << "\t" << w[it->second] << std::endl;
    }
    out.flush();
}

#endif
