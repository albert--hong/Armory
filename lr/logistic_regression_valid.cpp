
#include <iostream>
#include <map>

#include "importer.hpp"

class LabelPair {
	public:
		LabelPair(float l, float p): label(l), pred(p) {}
		float label;
		float pred;
		bool operator < (const LabelPair &pair) const {
			return pred < pair.pred;
		}
};

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
		std::cout << "Usage: " << argv[0] << " features_dict instances_file model_file" << std::endl;
        return -1;
    }
    const char* pathFeature = argv[1];
    const char* pathInstance = argv[2];
    const char* pathBaseModel = argv[3];
    std::map<std::string, uint32_t> features;
    std::vector<std::vector<uint32_t> > X;
    std::vector<uint8_t> y;
    std::map<std::string, float> baseModelWeights;
    int maxFeaIdx = 0;
    loadFeatures(pathFeature, features, maxFeaIdx);         // 加载特征集合
    loadInstances(pathInstance, X, y);                     	// 加载训练样本
	loadBaseModel(pathBaseModel, baseModelWeights);         // 加载基础模型

    // 增加 Bias 特征
    int baisFeaIdx = maxFeaIdx + 1;
    features.insert(std::make_pair("Bias", baisFeaIdx));
    for (int xi = 0; xi < X.size(); ++xi)
    {
        X[xi].push_back(baisFeaIdx);
    }
    std::vector<float> weights(features.size());
    for (std::map<std::string, uint32_t>::iterator it = features.begin(); it != features.end(); ++it) {
        std::string feaName = it->first;
        int feaIdx = it->second;
		std::map<std::string, float>::iterator weightItem =  baseModelWeights.find(feaName);
        if (weightItem != baseModelWeights.end()) {
            weights[feaIdx] = weightItem->second;
        } else {
            weights[feaIdx] = 0.0f;
        }
    }
	
	std::cout << "start training: feature size = " << features.size()
        << "; instances count = " << X.size()
        << "; create weights = " << weights.size() << std::endl;

	// 计算 AUC
	size_t posCount = 0;
	size_t negCount = 0;
	uint32_t sigmaRank = 0;
	std::vector<LabelPair> pairs;
	for (int xi = 0; xi < X.size(); ++xi) {
		float label = y[xi];
		float pred = sigmoid(X[xi], weights);
		LabelPair pair(label, pred);
		pairs.push_back(pair);
		if (label > 0.1) {
			posCount += 1;
		} else {
			negCount += 1;
		}
	}
	sort(pairs.begin(), pairs.end());

	for (size_t pi = 0; pi < pairs.size(); ++pi) {
		if (pairs[pi].label > 0.1) {
			sigmaRank += (pi + 1);
		}
	}
	std::cout << pairs.size() << "\t" << sigmaRank << "\t" << posCount << "\t" << negCount << std::endl;
	
	double AUC = static_cast<double>(sigmaRank - (posCount * (posCount + 1) / 2)) / static_cast<double>(posCount * negCount);
	printf("The AUC of lr model = %.4f\n", AUC);
}
