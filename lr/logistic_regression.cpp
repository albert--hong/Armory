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
#include <stdlib.h>

#include "importer.hpp"

using namespace std;
using std::cout;
using std::endl;

double sigmoid(std::set<int> &x, std::vector<double> &w)
{
    double z = 0.0;
    for (std::set<int>::iterator xi = x.begin(); xi != x.end(); xi++)
    {
        z += w[*xi];
    } double p = 1.0 / (1.0 + exp(-z)); return p;
}
/* 
 * 离散值 LR 模型的训练函数.
 * @param vector<set<int> > 训练样本特征 DataFrame
 * @param vector<double> 训练样本 label
 * @param vector<double> 模型参数
 */
void discreteLR(vector<set<int> > &X, vector<double> &y, vector<double> &w)
{
    // 停止条件1  loss <= epsilon
    double epsilon = 1e-3;
    // 停止条件2  iter < maxIters
    int maxIters = 100;
    // 学习率
    double gamma = 1;
    // 正则化系数 L1 系数
    double lambda1 = 0.00001;
    // 正则化系数 L2 系数
    double lambda2 = 0.0001;
    // 批次大小
    int batchSize = 100;
    // BGD
    for (int iter = 0; iter < maxIters; ++iter)
    {
        std::cout << "==============================================" << std::endl;
        std::cout << "training iter " << iter << " start..." << std::endl;
        size_t batchIter = 0;
        size_t xs = 0;
        size_t xe = xs + batchSize;
        while (xs < X.size() - 1)
        {
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
                    if (X[xi].find(wj) != X[xi].end())
                    {
                        predict_gd += (y_hat - y[xi]);
                    }
                }
                predict_gd = predict_gd / miniBatchSize;
                // 计算两个正则项的梯度
                double lambda1_gd = w[wj] > 0 ? lambda1 : -lambda1;
                double lambda2_gd = 2 * lambda2 * w[wj];
                double wj_gradient = predict_gd + lambda1_gd + lambda2_gd;
                // 更新梯度
                w[wj] = w[wj] - gamma * predict_gd;
            } // end wj
            xs = xe;
            xe = xe + batchSize;
        } // end batchIter
        // 一轮结束输出相关日志
        std::cout << "training iter " << iter << " finished." << std::endl;
        // 计算前 10000 个样本的 loss
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
        if (predictLoss + regularLoss < epsilon)
            break;
    }
}
/* 
 * 离散值 LR 模型的训练函数.
 * @param vector<set<int> > 训练样本特征 DataFrame
 * @param vector<double> 训练样本 label
 * @param vector<double> 模型参数
 */
void discreteLR(vector<set<int> > &X, vector<double> &y, vector<double> &w)
{
    // 停止条件1  loss <= epsilon
    double epsilon = 1e-3;
    // 停止条件2  iter < maxIters
    int maxIters = 100;
    // 学习率
    double gamma = 1;
    // 正则化系数 L1 系数
    double lambda1 = 0.00001;
    // 正则化系数 L2 系数
    double lambda2 = 0.0001;
    // 批次大小
    int batchSize = 100;
    // BGD
    for (int iter = 0; iter < maxIters; ++iter)
    {
        std::cout << "==============================================" << std::endl;
        std::cout << "training iter " << iter << " start..." << std::endl;
        size_t batchIter = 0;
        size_t xs = 0;
        size_t xe = xs + batchSize;
        while (xs < X.size() - 1)
        {
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
                    if (X[xi].find(wj) != X[xi].end())
                    {
                        predict_gd += (y_hat - y[xi]);
                    }
                }
                predict_gd = predict_gd / miniBatchSize;
                // 计算两个正则项的梯度
                double lambda1_gd = w[wj] > 0 ? lambda1 : -lambda1;
                double lambda2_gd = 2 * lambda2 * w[wj];
                double wj_gradient = predict_gd + lambda1_gd + lambda2_gd;
                // 更新梯度
                w[wj] = w[wj] - gamma * predict_gd;
            } // end wj
            xs = xe;
            xe = xe + batchSize;
        } // end batchIter
        // 一轮结束输出相关日志
        std::cout << "training iter " << iter << " finished." << std::endl;
        // 计算前 10000 个样本的 loss
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
        if (predictLoss + regularLoss < epsilon)
            break;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "Usage: " << argv[0] << " features_dict instances_file model_file" << std::endl;
        return -1;
    }
    std::map<std::string, int> features;
    std::vector<std::set<int> > X;
    std::vector<double> y;
    int maxFeaIdx = 0;
    loadFeatures(argv[1], features, maxFeaIdx);
    loadInstances(argv[2], X, y);

    // 增加 Bias 特征
    int baisFeaIdx = maxFeaIdx + 1;
    features.insert(std::make_pair("Bias", baisFeaIdx));
    for (int xi = 0; xi < X.size(); ++xi) {
        X[xi].insert(baisFeaIdx);
    }

    std::vector<double> weights;
    for (int i = 0; i < features.size(); ++i)
    {
        double randWeight = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        weights.push_back(randWeight);
    }
    cout << "start training: feature size = " << features.size()
         << "; instances count = " << X.size()
         << "; create weights = " << weights.size() << std::endl;
    discreteLR(X, y, weights);
    outputModel(argv[3], weights, features);
}
