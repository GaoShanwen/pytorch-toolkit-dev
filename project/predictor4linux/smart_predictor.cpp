// #include "ncnn/net.h"
// #include "smart_predictor.h"
// #include "opencv2/opencv.hpp"

#include "utils.h"
#include "index.h"
#include "classifierFactory.h"

#include "spdlog/spdlog.h"
#include <spdlog/cfg/env.h>  // support for loading levels from the environment  variable
#include <spdlog/fmt/ostr.h> // support for user defined types


using namespace std;
using namespace spdlog;
// using namespace cv;

shared_ptr<FlatIndex> g_index = NULL;
// shared_ptr<IClassifier> g_classifier = NULL;
IClassifier* g_classifier = NULL;

extern "C" int predictInit(const char* model_dir, const char* nx_dir, const int model_type)
{
    spdlog::info("SmartPredictor_init: {}", "init_camera");
    ClassifierFactory* factory = new ClassifierFactory();

    string tty_port = model_dir;
    spdlog::info("SmartPredictor_init: {}", tty_port);
    g_classifier = factory->create(tty_port);
    spdlog::info("SmartPredictor_init: {}", "init_camera_suc");
    delete factory;

    string nxname = nx_dir;
    nxname += "modelnew.nx";

    g_index = make_shared<FlatIndex>();
    spdlog::info("SmartPredictor_init: nxpath={}", nxname);

    ifstream f(nxname.c_str());
    spdlog::info("SmartPredictor_init: {}", "nxname is right");
    if (f.good()) {
        if (!g_index->decrypt_load(nxname)) {
            spdlog::info("SmartPredictor_init: {}", "error modelnew.nx load_nx-no-g_index->init_120000");
            if (g_index->init(120000, 128, 120000, 0)) return 4; else return -4;
        }
    } else {
        spdlog::error("SmartPredictor_init: {}", "no modelnew.nx load_nx-no-g_index->init_120000");
        if (g_index->init(120000, 128, 120000, 0)) return 3; else return -3;
    }
    spdlog::info("SmartPredictor_init: {}", "load_nx_suc");
    return 1;
}


extern "C" const char* predict(const char* image_path, const int limit) {
    string result = "";
    int nTopK = 5;
    int nFeaDim = 128;
    float *pfFea = new float[nFeaDim];
    spdlog::info("predict: start detect...");
    g_classifier->get_fea(pfFea, nFeaDim);

    // 3. search
    int *pnResIDIdx = new int[nTopK];
    float *pfResScoreIdx = new float[nTopK];
    int *indexID = new int[nTopK];
    memset(pnResIDIdx, 0, sizeof(int) * nTopK);
    memset(pfResScoreIdx, 0, sizeof(float) * nTopK);
    memset(indexID, 0, sizeof(int) * nTopK);

    auto t1 = GetCurrentTime();
    int oriK = 30;
    int finalK = 5;
    g_index->searchKnn(pfFea, nFeaDim, pnResIDIdx, pfResScoreIdx, indexID, limit, oriK, finalK);

    string result1;
    for (int i = 0; i < nTopK; i++) {
        result1.append(to_string(pnResIDIdx[i]));
        result1.append(",");
        result1.append(to_string(pfResScoreIdx[i]));
        result1.append(",");
        result1.append(to_string(indexID[i]));
        result1.append(",");
        result1.append(g_index->getOssKeyByIndex(indexID[i]));

        if (i != nTopK - 1) result1.append(";");
    }
    
    double predictTime1 = GetElapsedTime(t1);
    spdlog::info("predict: c++ search costs {} ms", predictTime1);
    spdlog::info("search-result: {}", result1.c_str());
    
    result.append(result1);

    delete[] pnResIDIdx;
    delete[] pfResScoreIdx;
    delete[] indexID;
    delete[] pfFea;
    result1.shrink_to_fit();
    return result.c_str();
}


extern "C" const char* registByImg(
    const char* input, int nClassID, const char* ossKey, int index, float limit
) {
    string result = "false";

    auto t = GetCurrentTime();
    float pfFea[1];
    int nClassNum = g_classifier->predict(pfFea, 0);
    double predictTime = GetElapsedTime(t);
    spdlog::info("registByImg: java detect costs {} ms", predictTime);

    if (nClassNum > 0) {
        // spdlog::info("registByImg: nClassNum");
        int nFeaDim = g_classifier->get_fea_dim();
        float *pfFea = new(std::nothrow) float[nFeaDim];
        if (pfFea == NULL) {
            // spdlog::info("registByImg, pfFea 分配内存失败！！");
            return result.c_str();
        }
        // spdlog::info("registByImg: get_fea");
        bool b = g_classifier->get_fea(pfFea, nFeaDim);

        auto rd1 = GetCurrentTime();
        bool redundancy_flag = g_index->check_similarity(pfFea, nFeaDim, nClassID);
        double predictTime1 = GetElapsedTime(rd1);
        spdlog::info("check_similarity cost: %f ms", predictTime1);

        if (b && !redundancy_flag) {
            // spdlog::info("registByImg: b");
            string ossKey_s = ossKey; // jstringToString(env, ossKey);
            spdlog::info("registByImg: ossKey_s={}", ossKey_s.c_str());
            char *ossKey_c = new(std::nothrow) char[150];
            if (ossKey_c == NULL) {
                // spdlog::info("registByImg, ossKey_c 分配内存失败！！");
                return result.c_str();
            }
            strcpy(ossKey_c, ossKey_s.c_str());
            //spdlog::info("registByImg: type1");
            ossKey_s.shrink_to_fit();

            auto t = GetCurrentTime();
            bool flag = g_index->insert(pfFea, nFeaDim, nClassID, ossKey_c, index, limit);
            result = flag ? "true" : "false";
            double predictTime = GetElapsedTime(t);
            spdlog::info("insert with data_clear costs %f ms", predictTime);
            delete[] ossKey_c;
        }
        delete[] pfFea;
        return result.c_str();
    }
    spdlog::info("predictOnly: nClassNum <= {}", 0);

    return result.c_str();
}

// extern "C" bool removeIndex(int index) {
//     g_index->removeIndex(index);
//     spdlog::info("removeIndex: index={d} \n", index);
//     return false;
// }

// int main() {
//     std::cout << "Running server..." << std::endl;
//     return 0;
// }
