#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <algorithm>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <thread>
#include <mutex>
#include <functional>
#include <chrono>

#include "importer.hpp"

using namespace std;
using std::cout;
using std::endl;

//////////////////// 配置项。 /////////////
// notes: 未来这些配置和方法都封装在类中 // 停止条件1  deltaLoss < epislon
const float epislon = 1e-5;
// 停止条件2  iter < maxIters
const int maxIters = 20;
// 学习率
const float learnRate = 0.01;
// 正则化系数 L1 系数
const float lambda1 = 0.01;
// 正则化系数 L2 系数
const float lambda2 = 0.01;
// 批次大小
const int batchSize = 128;
// 并行训练的线程数
const int threadCount = 30;

/** 
 * 基于 mini-batch SGD 优化过程。线程安全，可用于并行 SGD 更新.
 * @param vector<vector<uint32_t> > 训练样本列表
 * @param vector<uint8_t> 训练样本标签
 * @param vector<float> 模型参数列表
 * @param std::mutex 模型参数锁
 * @param int 最大循环次数
 */
void LrTrainSgd(vector<vector<uint32_t> > &X,
                      vector<uint8_t> &y,
                      vector<float> &w,
                      std::mutex &lockW,
                      int &status,
                      int miniBatchCountPerThread)
{
    for (uint32_t bi = 0; bi < miniBatchCountPerThread && status == 0; ++bi)
    {
        vector<float> deltaLoss(w.size());      // mini-batch 计算出来的梯度
        vector<float> regularLoss(w.size());    // 正则化损失
        vector<float> updateWeight(w.size());   // 梯度更新
        vector<float> varLoss(w.size());        // mini-batch 计算出来的 delta loss 的方差，用来判断早停条件
        // 随机获取一段训练数据，计算梯度
        uint32_t xs = rand() % (X.size() - batchSize);
        float stopCriterion = 0.0;
        for (int xi = xs; xi < xs + batchSize && xi < X.size(); ++xi)
        {
            float y_hat = sigmoid(X[xi], w);
            float y_hat_xi = (y_hat - y[xi]);
			for (int xj = 0; xj < X[xi].size(); ++xj) {
				uint32_t xij = X[xi][xj]; 
				if (xij < w.size()) {
					deltaLoss[xij] += y_hat_xi;
					varLoss[xij] += (y_hat_xi * y_hat_xi);
				}
			} // end xj
        } // end xi
        for (int wi = 0; wi < w.size(); wi++) {
            deltaLoss[wi] /= batchSize;
            varLoss[wi] /= (batchSize - 1);
			varLoss[wi] -= (deltaLoss[wi] * deltaLoss[wi]);
            stopCriterion += ((deltaLoss[wi] * deltaLoss[wi]) / varLoss[wi]);
            float l1Loss = w[wi] > 0 ? lambda1 : -lambda1;
            float l2Loss = 2 * lambda2 * w[wi];
            regularLoss[wi] = (l1Loss + l2Loss);
            updateWeight[wi] = -learnRate * (deltaLoss[wi] + regularLoss[wi]);
        } // end wi
        // 更新模型 weight
        lockW.lock();
        // 判断早停条件
        if (stopCriterion * batchSize < w.size()) {
            status = 1;
        }
        for (int wi = 0; wi < w.size(); wi++) {
            w[wi] += updateWeight[wi];
        }
        lockW.unlock();
    } // end mini batch
	lockW.lock();
	status = 1;
	lockW.unlock();
}

/**
 * 循环输出模型
 */
void outputModelLoop(const char *pathModel, std::vector<float> &w,
                 std::map<std::string, uint32_t> &features, int &status)
{
	while (status == 0) {
		ofstream out(pathModel);
		for (std::map<std::string, uint32_t>::iterator it = features.begin(); it != features.end(); ++it)
		{
			out << it->first << "\t" << w[it->second] << std::endl;
		}
		out.flush();
		out.close();
		sleep(60 * 0.1); 
		//sleep(60 * 10); 
	}
}

/**
 * 离散值 LR 模型的训练函数。用最传统的 Batch Gradient Descent 训练方式
 * 串行版本，即将下线。
 * @param vector<set<int> > 训练样本特征 DataFrame
 * @param vector<double> 训练样本 label
 * @param vector<double> 模型参数
 */
void discreteLR(vector<vector<uint32_t> > &X, vector<uint8_t> &y, vector<float> &w, 
        std::map<std::string, uint32_t> &features, const char* pathModel)
{

    // Batch Gradient Desent
    for (int iter = 0; iter < maxIters; ++iter)
    {
        std::cout << "==============================================" << std::endl;
        std::cout << "training iter " << iter << " start..." << std::endl;
        size_t batchIter = 0;
        size_t xs = 0;
        size_t xe = xs + batchSize;
        double deltaLoss = 0.0;
        double avgDeltaLoss = 0.0;
        size_t avgDeltaLossWindow = 100;
        while (xs < X.size() - 1)
        {
            batchIter ++;
            if (xe >= X.size())
            {
                xe = X.size();
            }
            int miniBatchSize = xe - xs;
            if (miniBatchSize <= 0)
            {
                break;
            }
            for (int wj = 0; wj < w.size(); ++wj)
            {
                // 计算 mini batch 的梯度
                double predict_gd = 0.0;
                for (int xi = xs; xi < xe; ++xi)
                {
                    double y_hat = sigmoid(X[xi], w);
                    predict_gd += (y_hat - y[xi]);
                }
                predict_gd = predict_gd / miniBatchSize;
                // 计算两个正则项的梯度
                double lambda1_gd = w[wj] > 0 ? lambda1 : -lambda1;
                double lambda2_gd = 2 * lambda2 * w[wj];
                double wj_gradient = predict_gd + lambda1_gd + lambda2_gd;
                // 更新梯度
                w[wj] = w[wj] - learnRate * wj_gradient;
                deltaLoss += (wj_gradient > 0 ? wj_gradient : -wj_gradient);
            } // end wj
            xs = xe;
            xe = xe + batchSize;
            deltaLoss = deltaLoss / static_cast<double>(w.size());
            avgDeltaLoss += deltaLoss;
            if (batchIter % avgDeltaLossWindow == 0)
            {
                std::cout << "training minibatch done. average delta loss = " << (avgDeltaLoss / avgDeltaLossWindow) << std::endl;
                avgDeltaLoss = 0.0;
            }
            if (deltaLoss < epislon)
            {
                return;
            }
            deltaLoss = 0.0;
        } // end batchIter
        // 一轮结束输出相关日志
        std::cout << "training iter " << iter << " finished." << std::endl;
        outputModel(pathModel, w, features);
    }
    // 退出前计算所有样本的Loss
    double predictLoss = 0.0;
    size_t predictCount = X.size();
    for (int xi = 0; xi < predictCount; ++xi)
    {
        double y_hat = sigmoid(X[xi], w);
        predictLoss = predictLoss - y[xi] * log(y_hat) - (1 - y[xi]) * log(1 - y_hat);
    }
    predictLoss = predictLoss / predictCount;
    double regularLoss = 0.0;
    for (int wj = 0; wj < w.size(); ++wj)
    {
        regularLoss += (lambda1 * (w[wj] > 0 ? w[wj] : -w[wj]));
        regularLoss += (lambda2 * w[wj] * w[wj]);
    }
    std::cout << "total loss = " << predictLoss + regularLoss
        << "; predict loss = " << predictLoss
        << "; regular loss = " << regularLoss << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cout << "Usage: " << argv[0] << " features_dict instances_file last_model_file new_model_file" << std::endl;
        return -1;
    }
    const char* pathFeature = argv[1];
    const char* pathInstance = argv[2];
    const char* pathBaseModel = argv[3];
    const char* pathModel = argv[4];
    std::map<std::string, uint32_t> features;
    std::vector<std::vector<uint32_t> > X;
    std::vector<uint8_t> y;
    std::map<std::string, float> baseModelWeights;
    int maxFeaIdx = 0;
    loadFeatures(pathFeature, features, maxFeaIdx);         // 加载特征集合
    loadInstances(pathInstance, X, y);                     	// 加载训练样本
	loadBaseModel(pathBaseModel, baseModelWeights);         // 加载基础模型

	cout << "load train data finished: feature size = " << features.size()
        << "; instances count = " << X.size() << endl;;

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
		map<std::string, float>::iterator weightItem =  baseModelWeights.find(feaName);
        if (weightItem != baseModelWeights.end()) {
            weights[feaIdx] = weightItem->second;
        } else {
            weights[feaIdx] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        }
    }
    cout << "add bias feature and create weights succeed. feature size = " << features.size()
        << "; create weights = " << weights.size() << std::endl;
    //discreteLR(X, y, weights, features, pathModel);
    size_t miniBatchCountPerThread = (maxIters * X.size()) / (batchSize * threadCount); // 每个线程的最大循环轮次
	cout << "miniBatchCountPerThread: " << miniBatchCountPerThread << endl;
    std::vector<std::thread*> threadPool;                                  // 初始化线程
    int status = 0;
    std::mutex lockW;
    for (int ti = 0; ti < threadCount; ti++) {
		std::thread* trainThread = new std::thread(LrTrainSgd, std::ref(X), std::ref(y),
                                         std::ref(weights), std::ref(lockW), std::ref(status), miniBatchCountPerThread);
		threadPool.push_back(trainThread);
    }
	std::thread outputThread(outputModelLoop, pathModel, std::ref(weights), 
			std::ref(features), std::ref(status));
    for (int ti = 0; ti < threadCount; ti++) {
        threadPool[ti]->join();
		delete threadPool[ti];
		threadPool[ti] = nullptr;
	}
	outputThread.join();
	outputModel(pathModel, weights, features);
}
