#include "index.h"
#include "spdlog/spdlog.h"
#include <spdlog/cfg/env.h>  // support for loading levels from the environment  variable
#include <spdlog/fmt/ostr.h> // support for user defined types

using namespace spdlog;
using namespace std;

FlatIndex::FlatIndex() {
    spdlog::info("index-conStrus");
    m_nListLen = 120000;
    m_nFeaLen = 128;
    m_OssLength = 150;
    m_nMaxClassID = 120000;
    m_nInsertNum = 0;

    m_pfFeaMat = NULL;
    m_pnClsIDMap = NULL;
    m_pnClsIDMapSearchTimes = NULL;
    m_pnClsIDMapSelectTimes = NULL;
    m_pnClsIDMapSameTimes = NULL;
    m_pnClsIDMapReplaceTimes = NULL;

    m_pForbiddens = NULL;
    m_pnClsIDOss = NULL;

    m_nMaxLoadTimes = 2000000;
    m_nMaxSearchTimes = 5000000;
    m_nMaxRegistTimes = 5000000;

    m_nLoadTimes = 0;
    m_nSearchTimes = 0;
    m_nRegistTimes = 0;

    save_index = 0;
    spdlog::info("index-conStrus-suc");
}


FlatIndex::~FlatIndex() {
    release();
}


void FlatIndex::release() {
    if (m_pfFeaMat != NULL) {
        delete[] m_pfFeaMat;
        m_pfFeaMat = NULL;
    }
    if (m_pnClsIDMap != NULL) {
        delete[] m_pnClsIDMap;
        m_pnClsIDMap = NULL;
    }

    if (m_pnClsIDMapSearchTimes != NULL) {
        delete[] m_pnClsIDMapSearchTimes;
        m_pnClsIDMapSearchTimes = NULL;
    }

    if (m_pnClsIDMapSelectTimes != NULL) {
        delete[] m_pnClsIDMapSelectTimes;
        m_pnClsIDMapSelectTimes = NULL;
    }

    if (m_pnClsIDMapSameTimes != NULL) {
        delete[] m_pnClsIDMapSameTimes;
        m_pnClsIDMapSameTimes = NULL;
    }

    if (m_pnClsIDMapReplaceTimes != NULL) {
        delete[] m_pnClsIDMapReplaceTimes;
        m_pnClsIDMapReplaceTimes = NULL;
    }

    if (m_pForbiddens != NULL) {
        delete[] m_pForbiddens;
        m_pForbiddens = NULL;
    }

    if (m_pnClsIDOss != NULL) {
        delete[] m_pnClsIDOss;
        m_pnClsIDOss = NULL;
    }

}


bool FlatIndex::init(int nListLen, int nFeaLen, int nMaxClassID, int nInsertNum) {
    m_nListLen = nListLen;
    m_nFeaLen = nFeaLen;
    m_nMaxClassID = nMaxClassID;
    m_nInsertNum = nInsertNum;
    spdlog::info("index-init: {} {} {} {} ", m_nListLen, m_nFeaLen, m_nMaxClassID, nInsertNum);


    spdlog::info("m_pfFeaMat, 分配内存开始");
    m_pfFeaMat = new(std::nothrow) float[m_nFeaLen * m_nListLen];
    if (m_pfFeaMat == NULL) {
        spdlog::info("m_pfFeaMat, 分配内存失败");
        return false;
    }
    spdlog::info("m_pfFeaMat, 分配内存成功! ");

    spdlog::info("m_pnClsIDMap, 分配内存开始");
    m_pnClsIDMap = new(std::nothrow) int[m_nListLen];
    if (m_pnClsIDMap == NULL) {
        spdlog::info("m_pnClsIDMap, 分配内存失败");
        return false;
    }
    spdlog::info("m_pnClsIDMap, 分配内存成功! ");
    memset(m_pnClsIDMap, 0, sizeof(int) * m_nListLen);

    m_pnClsIDMapSearchTimes = new(std::nothrow) int[m_nListLen];
    if (m_pnClsIDMapSearchTimes == NULL) {
        spdlog::info("m_pnClsIDMapSearchTimes, 分配内存失败");
        return false;
    }
    spdlog::info("m_pnClsIDMapSearchTimes, 分配内存成功! ");
    memset(m_pnClsIDMapSearchTimes, 0, sizeof(int) * m_nListLen);

    m_pnClsIDMapSelectTimes = new(std::nothrow) int[m_nListLen];
    if (m_pnClsIDMapSelectTimes == NULL) {
        spdlog::info("m_pnClsIDMapSelectTimes, 分配内存失败");
        return false;
    }
    spdlog::info("m_pnClsIDMapSelectTimes, 分配内存成功! ");
    memset(m_pnClsIDMapSelectTimes, 0, sizeof(int) * m_nListLen);

    m_pnClsIDMapSameTimes = new(std::nothrow) int[m_nListLen];
    if (m_pnClsIDMapSameTimes == NULL) {
        spdlog::info("m_pnClsIDMapSameTimes, 分配内存失败");
        return false;
    }
    spdlog::info("m_pnClsIDMapSameTimes, 分配内存成功! ");
    memset(m_pnClsIDMapSameTimes, 0, sizeof(int) * m_nListLen);

    m_pnClsIDMapReplaceTimes = new(std::nothrow) int[m_nListLen];
    if (m_pnClsIDMapReplaceTimes == NULL) {
        spdlog::info("m_pnClsIDMapReplaceTimes, 分配内存失败");
        return false;
    }
    spdlog::info("m_pnClsIDMapReplaceTimes, 分配内存成功! ");
    memset(m_pnClsIDMapReplaceTimes, 0, sizeof(int) * m_nListLen);


    spdlog::info("m_pnClsIDOss, 分配内存开始");
    m_pnClsIDOss = new(std::nothrow) char[m_OssLength * m_nListLen];
    if (m_pnClsIDOss == NULL) {
        spdlog::info("m_pnClsIDOss, 分配内存失败");
        return false;
    }
    spdlog::info("m_pnClsIDOss, 分配内存成功! {} ", sizeof(char));
    memset(m_pnClsIDOss, 0, sizeof(char) * m_OssLength * m_nListLen);

    spdlog::info("init() suc!");
    return true;
}


bool FlatIndex::vCompare(const pair<int, float> p1, const pair<int, float> p2) {
    return p1.second > p2.second;
}

bool FlatIndex::vCompareKnn(const pair<int, pair<int, float>> p1, const pair<int, pair<int, float>> p2) {
    return p1.second.second > p2.second.second;
}

bool FlatIndex::vCompareEvery(
    const pair<int, pair<float, pair<int, float> *>> p1, const pair<int, pair<float, pair<int, float> *>> p2)
{
    return p1.second.first > p2.second.first;
}

bool FlatIndex::vCompareInt(const pair<int, int> p1, const pair<int, int> p2) {
    return p1.second > p2.second;
}

inline bool FlatIndex::normalize_l2(float *pfFea, int nDim) {
    float fSquareSum = 0;
    for (int i = 0; i < nDim; i++) {
        fSquareSum += pfFea[i] * pfFea[i];
    }
    if (fSquareSum == 0) {
        return true;
    }
    float fL2Norm = sqrt(fSquareSum);
    for (int i = 0; i < nDim; i++) {
        pfFea[i] /= fL2Norm;
    }

    return true;
}


inline bool FlatIndex::softmax(float *scores, int sSize) {
    float max_val = scores[0];
    vector<float> e_x(sSize);
    for (int i = 0; i < sSize; i++) {
        e_x[i] = exp(scores[i] - scores[0]);
    }
    float sum = 0.0;
    for (int i = 0; i < sSize; i++)
        sum += e_x[i];

    for (int i = 0; i < sSize; i++) {
        scores[i] = e_x[i] / sum;
    }
    return true;
}


inline float FlatIndex::similarity(const float *pfFea1, const float *pfFea2, int nDim) {
    float fSim = 0;
    for (int i = 0; i < nDim; i++) {
        fSim += pfFea1[i] * pfFea2[i];
    }

    return fSim;
}


bool FlatIndex::insert(const float *pfFea, int nDim, int nClassID, char *ossKey, int index, float limit) {
    // step 1: check classID, feature dimension and normalize
    spdlog::info("index-insert: nClassID({0}) m_nInsertNum({1}) {2} {3} {4} {5} {6}", 
        nClassID, m_nInsertNum, index, limit, ossKey, nDim, m_nFeaLen);

    if (nDim != m_nFeaLen) {
        return false;
    }

    float *pfFeaNorm = new(std::nothrow) float[m_nFeaLen];
    if (pfFeaNorm == NULL) {
        spdlog::info("insert, pfFeaNorm 分配内存失败! ! ");
        return false;
    }
    memcpy(pfFeaNorm, pfFea, sizeof(float) * m_nFeaLen);
    normalize_l2(pfFeaNorm, m_nFeaLen);

    vector<pair<int, float>> vecCls;
    vector<pair<int, int>> vecPos;
    const int dataClearGap = 20000;
    const float setThreshold = 0.6;
    spdlog::info("data_Clear gallery data number={d} ", m_nInsertNum);
    if (m_nInsertNum > dataClearGap) {
        int len = min(m_nListLen, m_nInsertNum);
        for (int i = 0; i < len; i++) {
            float fSim = similarity(pfFeaNorm, m_pfFeaMat + m_nFeaLen * i, m_nFeaLen);
            if (fSim > setThreshold) {
                vecCls.push_back(pair<int, float>(m_pnClsIDMap[i], fSim));
                vecPos.push_back(pair<int, int>(i, fSim));
            }
        }
        sort(vecCls.begin(), vecCls.end(), vCompare);
        sort(vecPos.begin(), vecPos.end(), vCompare);
        int nRes = vecCls.size();
        int nVoteRange = min(30, nRes);
        spdlog::info("data_Clear nVoteRange number={0} setThreshold={1}", nVoteRange, setThreshold);

        vector<pair<int, int>> pnResID;
        for (int i = 0; i < nVoteRange; i++) {
            if (vecCls[i].first != nClassID) continue;
            m_pnClsIDMapSelectTimes[vecPos[i].first] += 1;
            spdlog::info("data_Clear Cls={d} Pos={d} Val={d}", vecCls[i].first, vecPos[i].first, m_pnClsIDMapSelectTimes[vecPos[i].first]);
        }
        vector<pair<int, float>>().swap(vecCls);
        vector<pair<int, int>>().swap(vecPos);
        int insertPos = m_nInsertNum % m_nListLen;
        if (m_nInsertNum > m_nListLen && m_pnClsIDMapSelectTimes[insertPos] != 0) {
            int choicePos = insertPos;
            for (int i=0; i<dataClearGap; i++) {
                int checkPos = (i + insertPos) % m_nListLen;
                if (m_pnClsIDMapSelectTimes[choicePos] > m_pnClsIDMapSelectTimes[checkPos]) {
                    choicePos = checkPos;
                    if (m_pnClsIDMapSelectTimes[choicePos] == 0)
                        break;
                }
            }
            spdlog::info("data_Clear choicePos-[{d}] insertPos-[{d}]", choicePos, insertPos);
            if (choicePos != insertPos) {
                memcpy(m_pfFeaMat + m_nFeaLen * choicePos, m_pfFeaMat + m_nFeaLen * insertPos, sizeof(float) * m_nFeaLen);
                memcpy(m_pnClsIDOss + m_OssLength * choicePos, m_pnClsIDOss + m_OssLength * insertPos, sizeof(char) * m_OssLength);

                m_pnClsIDMap[choicePos] = m_pnClsIDMap[insertPos];
                m_pnClsIDMapSelectTimes[choicePos] = m_pnClsIDMapSelectTimes[insertPos];
            }

        }
        else {
            spdlog::info("data_Clear insertPos-[{d}] passed", insertPos);
        }
    }

    int cn = 0;
    if (cn == 0) {

        int nPos = m_nInsertNum % m_nListLen;
        memcpy(m_pfFeaMat + m_nFeaLen * nPos, pfFeaNorm, sizeof(float) * m_nFeaLen);
        memcpy(m_pnClsIDOss + m_OssLength * nPos, ossKey, sizeof(char) * m_OssLength);

        m_pnClsIDMap[nPos] = nClassID;
        m_pnClsIDMapSelectTimes[nPos] = 0;
        m_nInsertNum++;
        spdlog::info("index-insert suc m_nInsertNum({d}) nPos({d}) ", m_nInsertNum, nPos);

    }

    delete[] pfFeaNorm;
    delete[] ossKey;

    return true;
}


bool FlatIndex::check_similarity(const float *pfFea, int m_nFeaLen, int setClassID) {
    float *pfFeaNorm = new(std::nothrow) float[m_nFeaLen];
    if (pfFeaNorm == NULL) {
        spdlog::info("search, pfFeaNorm 分配内存失败! ! ");
        return false;
    }
    memcpy(pfFeaNorm, pfFea, sizeof(float) * m_nFeaLen);
    normalize_l2(pfFeaNorm, m_nFeaLen);

    int setClassIDCount = 0;
    float max_scores = 0.0;
    float similary_th = 0.99;
    bool redundancy_flag = false;
    for (int i = 0; i < min(m_nListLen, m_nInsertNum); i++) {
        int nClassID = m_pnClsIDMap[i];
        if (nClassID != setClassID) {
            continue;
        }
        float fSim = similarity(pfFeaNorm, m_pfFeaMat + m_nFeaLen * i, m_nFeaLen);
        max_scores = max(max_scores, fSim);
        setClassIDCount++;
        similary_th = 0.99 - 0.01 * (min(setClassIDCount/20, 1) + 3 * min(setClassIDCount/100, 1) \
                            + 5 * min(setClassIDCount/500, 1));
        if (max_scores > similary_th) {
            redundancy_flag = true;
            break;
        }
    }
    int b = redundancy_flag;
    spdlog::info("check_similarity: nClassID({0}) m_nInsertNum({1}+) {2} {3}", setClassID, setClassIDCount, similary_th, b);
    return redundancy_flag;
}


bool FlatIndex::removeIndex(int index) {
    m_pnClsIDMap[index] = -1;
    return true;
}


string FlatIndex::getOssKeyByIndex(int index) {
    if (index > 0) {
        return m_pnClsIDOss + m_OssLength * index;
    } else {
        return "";
    }

}


bool FlatIndex::getIndexByClassID(int nClassIDp, int pageno, int pageSize, int *indexID) {
    int j = 0;
    for (int i = 0; i < min(m_nListLen, m_nInsertNum); i++) {
        int nClassID = m_pnClsIDMap[i];
        if (nClassID == nClassIDp) {
            if (j >= (pageno - 1) * pageSize && j < pageno * pageSize) {
                indexID[j] = i;
            }
            j++;
        }
        if (j >= pageno * pageSize) {
            break;
        }
    }

    return true;
}


bool
FlatIndex::searchKnn(const float *pfFea, int nDim, int *pnResID, float *pfResScore,
                     int *indexID, float limit, int oriK, int finalK) {

    m_nSearchTimes += 1;
    if (nDim != m_nFeaLen) {
        return false;
    }
    float *pfFeaNorm = new(std::nothrow) float[m_nFeaLen];
    if (pfFeaNorm == NULL) {
        spdlog::info("searchKnn, pfFeaNorm 分配内存失败! ! ");
        return false;
    }
    memcpy(pfFeaNorm, pfFea, sizeof(float) * m_nFeaLen);
    normalize_l2(pfFeaNorm, m_nFeaLen);

    for (int i = 0; i < 10; i++) {
        spdlog::info("before-search feat[{0}]: {1} {2}", i, pfFea[i], pfFeaNorm[i]);
    }
    int lengthIDs = 5;
    for (int i = 0; i < 5; i++) {
        spdlog::info("before-search[{0}], {1} {2} ", i, pnResID[i], m_nMaxClassID);
        if (pnResID[i] == 0) {
            lengthIDs = i;
            break;
        }
    }
    if (m_pForbiddens == NULL)
        spdlog::info("before-search m_pForbiddens is NULL !");
    vector<pair<int, float> > vecRes;
    for (int i = 0; i < min(m_nListLen, m_nInsertNum); i++) {
        int nClassID = m_pnClsIDMap[i];
        bool passFlag = false;
        for (int j=0; j<lengthIDs; j++){
            if (nClassID == pnResID[j]) {
                passFlag = true;
                break;
            }
        }
        if (nClassID == -1 or passFlag or (m_pForbiddens != NULL and m_pForbiddens[i])) {
//            spdlog::info("continue index ：{d} ", i);
            continue;
        }
//        if (m_pForbiddens != NULL and m_pForbiddens[i]) {
//            spdlog::info("search continue index ：{d} clsID={d} ", i, nClassID);
//            continue;
//        }
//        else if (m_pForbiddens) {
//            spdlog::info("search continue index ：{d} clsID={d} m_pForbiddens=False", i, nClassID);
//        }
        float fSim = similarity(pfFeaNorm, m_pfFeaMat + m_nFeaLen * i, m_nFeaLen);
        if (fSim > limit) {
            vecRes.push_back(pair<int, float>(nClassID, fSim));
        }
    }

    sort(vecRes.begin(), vecRes.end(), vCompare);

    int nRes = vecRes.size();
    int nVoteRange = min(oriK, nRes);

    vector<pair<int, float>> vecRes2;

    for (int i = 0; i < nVoteRange; i++) {

        int bhava = 0;
        for (int j = 0; j < vecRes2.size(); j++) {

            if (vecRes[i].first == vecRes2[j].first) {
                vecRes2[j].second = vecRes2[j].second + vecRes[i].second;
                bhava = 1;
            }
        }
        if (bhava == 0) {
            vecRes2.push_back(vecRes[i]);
        }

    }

    sort(vecRes2.begin(), vecRes2.end(), vCompare);

    int size2 = vecRes2.size();
    int size2f = min(finalK, size2);
    int lengthRes = max(size2f, lengthIDs);

    for (int i = 0; i < lengthRes; i++) {
        pnResID[i] = i >= size2f ? 0:vecRes2[i].first;
        pfResScore[i] = i >= size2f ? 0.:vecRes2[i].second;
        spdlog::info("search-knn[{d}], {d} {d}, %f ", i, size2f, pnResID[i], pfResScore[i]);
    }
    pnResID[lengthRes] = 0;


    delete[] pfFeaNorm;
    vector<pair<int, float>>().swap(vecRes2);
    vector<pair<int, float>>().swap(vecRes);

    return true;
}


bool FlatIndex::train(float limit) {

    spdlog::info("train-start , %f ", limit);

    int lastSize = 0;
    int *already_IDs = new(std::nothrow) int[m_nListLen];
    if (already_IDs == NULL) {
        spdlog::info("train, already_IDs 分配内存失败! ! ");
        return false;
    }
    memset(already_IDs, 0, sizeof(int) * m_nListLen);


    for (int i = 0; i < min(m_nListLen, m_nInsertNum); i++) {
        int nClassID = m_pnClsIDMap[i];
        if (nClassID == -1) {
            //spdlog::info("continue index ：{d} ", i);
            continue;
        }
        vector<pair<int, int> > vecRes;
        vecRes.push_back(pair<int, int>(nClassID, i));
        for (int k = i + 1; k < min(m_nListLen, m_nInsertNum); k++) {
            int nClassID = m_pnClsIDMap[i];
            if (nClassID == -1) {
                //spdlog::info("train-continue -1 ：{d} ", i);
                continue;
            }
            int b_hava = 0;
            for (int j = 0; j < lastSize; j++) {
                if (nClassID == already_IDs[j]) {
                    b_hava = 1;
                    break;
                }
            }
            if (b_hava == 1) {
                //spdlog::info("train-continue b_hava ：{d} ", i);
                continue;
            }

            float fSim = similarity(m_pfFeaMat + m_nFeaLen * i, m_pfFeaMat + m_nFeaLen * k,
                                    m_nFeaLen);
            if (fSim > limit) {
                vecRes.push_back(pair<int, int>(nClassID, k));
            }
        }

        int nRes = vecRes.size();
        if (nRes <= 5) {
            //spdlog::info("train-continue size <{d} ", nRes);
            continue;
        }

        vector<pair<int, int>> vecRes2;

        for (int i = 0; i < nRes; i++) {

            int bhava = 0;
            for (int j = 0; j < vecRes2.size(); j++) {

                if (vecRes[i].first == vecRes2[j].first) {
                    vecRes2[j].second = vecRes2[j].second + 1;
                    bhava = 1;
                }
            }
            if (bhava == 0) {
                vecRes2.push_back(vecRes[i]);
            }

        }

        sort(vecRes2.begin(), vecRes2.end(), vCompareInt);

        int nClassIDReal = vecRes2[0].first;

        lastSize = nRes;
        memset(already_IDs, 0, sizeof(int) * m_nListLen);

        for (int i = 0; i < nRes; i++) {

            already_IDs[i] = vecRes[i].second;

            if (m_pnClsIDMap[vecRes[i].second] != nClassIDReal) {
                int nClassIDOld = m_pnClsIDMap[vecRes[i].second];
                m_pnClsIDMap[vecRes[i].second] = nClassIDReal;
                //spdlog::info("train-replace , {d}, {d}, {d} ", vecRes[i].second, nClassIDOld, nClassIDReal);
            }
        }

        vector<pair<int, int>>().swap(vecRes2);
        vector<pair<int, int>>().swap(vecRes);

    }


    return true;
}


bool FlatIndex::index2buffer(unsigned char *pszDataBuf, int nDataLen) {
    int nOffset = 0;
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nListLen, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nFeaLen, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nMaxClassID, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nInsertNum, sizeof(int));
    nOffset += sizeof(int);

    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nMaxLoadTimes, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nMaxSearchTimes, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nMaxRegistTimes, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nLoadTimes, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nSearchTimes, sizeof(int));
    nOffset += sizeof(int);
    memcpy(pszDataBuf + nOffset, (unsigned char *) &m_nRegistTimes, sizeof(int));
    nOffset += sizeof(int);

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pfFeaMat,
           sizeof(float) * m_nFeaLen * m_nListLen);
    nOffset += sizeof(float) * m_nFeaLen * m_nListLen;

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pnClsIDMap, sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;

    std::cout << "index2buffer() 1" << std::endl;

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pnClsIDMapSearchTimes,
           sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;

    std::cout << "index2buffer() 2" << std::endl;

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pnClsIDMapSelectTimes,
           sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;

    std::cout << "index2buffer() 3" << std::endl;

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pnClsIDMapSameTimes,
           sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;

    std::cout << "index2buffer() 4" << std::endl;

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pnClsIDMapReplaceTimes,
           sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;

    std::cout << "index2buffer() 5" << std::endl;

    memcpy(pszDataBuf + nOffset, (unsigned char *) m_pnClsIDOss,
           sizeof(char) * m_OssLength * m_nListLen);
    nOffset += sizeof(char) * m_OssLength * m_nListLen;

    std::cout << "index2buffer() suc" << std::endl;

    return true;
}


bool FlatIndex::buffer2index(unsigned char *pszDataBuf, int nDataLen) {
    release();
    int nOffset = 0;
    memcpy((unsigned char *) &m_nListLen, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nFeaLen, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxClassID, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);

    if (m_nListLen != 120000 && m_nFeaLen != 128 && m_nMaxClassID != 120000) {
        spdlog::info("buffer2index() jie xi errr: {0} {1} {2} ", m_nListLen, m_nFeaLen, m_nMaxClassID);
        return false;
    }

    if (nDataLen != 81840040) {
        spdlog::info("buffer2index() size errr: {} ", nDataLen);
        return false;
    }

    memcpy((unsigned char *) &m_nInsertNum, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);

    if (m_nInsertNum < 0 || m_nInsertNum > 10000000) {
        spdlog::info("buffer2index() m_nInsertNum errr: {d} ", m_nInsertNum);
        return false;
    }

    spdlog::info("buffer2index-2: {0} {1} {2} {3} ", m_nListLen, m_nFeaLen, m_nMaxClassID, m_nInsertNum);

    bool b = init(m_nListLen, m_nFeaLen, m_nMaxClassID, m_nInsertNum);
    if (!b) {
        spdlog::info("buffer2index, init error! ");
        return false;
    }
    spdlog::info("buffer2index, init suc! ");

    memcpy((unsigned char *) &m_nMaxLoadTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxSearchTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nMaxRegistTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nLoadTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nSearchTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    memcpy((unsigned char *) &m_nRegistTimes, pszDataBuf + nOffset, sizeof(int));
    nOffset += sizeof(int);
    spdlog::info("buffer2index, memcpy1 nOffset={}", nOffset);
    spdlog::info("buffer2index, memcpy1.1 {0} {1} {2} {3} {4} {5}",
         m_nMaxLoadTimes, m_nMaxSearchTimes, m_nMaxRegistTimes, m_nLoadTimes, m_nSearchTimes, m_nRegistTimes);
    memcpy((unsigned char *) m_pfFeaMat, pszDataBuf + nOffset,
           sizeof(float) * m_nFeaLen * m_nListLen);
    nOffset += sizeof(float) * m_nFeaLen * m_nListLen;
    spdlog::info("buffer2index, memcpy2  nOffset={}", nOffset);
    memcpy((unsigned char *) m_pnClsIDMap, pszDataBuf + nOffset, sizeof(int) * m_nListLen);
    nOffset += sizeof(int) * m_nListLen;
    spdlog::info("buffer2index, memcpy3  nOffset={}", nOffset);
    spdlog::info("buffer2index, memcpy3.1  m_pnClsID[0]={0}, m_pnClsID[1]={1}", m_pnClsIDMap[0], m_pnClsIDMap[1]);

    memcpy((unsigned char *) m_pnClsIDMapSearchTimes, pszDataBuf + nOffset,
           sizeof(int) * m_nListLen);
    // std::cout << "buffer2index() 4" << std::endl;
    nOffset += sizeof(int) * m_nListLen;
    spdlog::info("buffer2index, memcpy4  nOffset={}", nOffset);

    memcpy((unsigned char *) m_pnClsIDMapSelectTimes, pszDataBuf + nOffset,
           sizeof(int) * m_nListLen);
    // std::cout << "buffer2index() 5" << std::endl;
    nOffset += sizeof(int) * m_nListLen;
    spdlog::info("buffer2index, memcpy5  nOffset={}", nOffset);

    memcpy((unsigned char *) m_pnClsIDMapSameTimes, pszDataBuf + nOffset,
           sizeof(int) * m_nListLen);
    // std::cout << "buffer2index() 6" << std::endl;
//    nOffset += sizeof(long int) * m_nListLen;
    nOffset += sizeof(int) * m_nListLen;
    spdlog::info("buffer2index, memcpy6  nOffset={}", nOffset);

    memcpy((unsigned char *) m_pnClsIDMapReplaceTimes, pszDataBuf + nOffset,
           sizeof(int) * m_nListLen);
    // std::cout << "buffer2index() 7" << std::endl;
//    nOffset += sizeof(long int) * m_nListLen;
    nOffset += sizeof(int) * m_nListLen;
    spdlog::info("buffer2index, memcpy7  nOffset={}", nOffset);
    // spdlog::info("buffer2index, memcpy7.1  m_pnClsIDMapReplaceTimes[0]={}", m_pnClsIDMapReplaceTimes);

    int sSize = sizeof(char) * m_OssLength * m_nListLen;
    spdlog::info("buffer2index, m_OssLength={0} size={1}", m_OssLength, sSize);
    memcpy((unsigned char *) m_pnClsIDOss, pszDataBuf + nOffset,
           sizeof(char) * m_OssLength * m_nListLen);
    // std::cout << "buffer2index() 8" << std::endl;
    nOffset += sizeof(char) * m_OssLength * m_nListLen;
    // spdlog::info("buffer2index, memcpy8  nOffset={d}", nOffset);
    string m_pnClsID0 = "";
    for (int i=0; i<150; i++) {
        m_pnClsID0 += m_pnClsIDOss[i];
    }
    spdlog::info("buffer2index, memcpy8.1  m_pnClsIDOss[0]={}", m_pnClsID0.c_str());

    if (m_pnClsIDMapReplaceTimes[m_nListLen - 1] < 0 ||
        m_pnClsIDMapReplaceTimes[m_nListLen - 2] < 0 ||
        m_pnClsIDMapReplaceTimes[m_nListLen - 3] < 0) {
        spdlog::info("buffer2index() ReplaceTimes last index errr: {0} {1} {2} ",
             m_pnClsIDMapReplaceTimes[m_nListLen - 1], m_pnClsIDMapReplaceTimes[m_nListLen - 2],
             m_pnClsIDMapReplaceTimes[m_nListLen - 3]);
        return false;
    }

    spdlog::info("buffer2index, 成功! ");
    return true;
}


bool FlatIndex::save(string strFileName) {
    int nDataLen = 0;
    nDataLen += sizeof(int) * 10;
    nDataLen += sizeof(float) * m_nFeaLen * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(char) * m_OssLength * m_nListLen;

    unsigned char *pszDataBuf = new(std::nothrow) unsigned char[nDataLen];
    if (pszDataBuf == NULL) {
        spdlog::info("save, pszDataBuf 分配内存失败! ! ");
        return false;
    }

    if (index2buffer(pszDataBuf, nDataLen) != true) {
        return false;
    }

    ofstream fout;
    fout.open(strFileName.c_str(), ios::binary);
    if (!fout.is_open()) {
        cerr << "failed to open the file:" << strFileName << " for writing." << endl;
        return false;
    }
    fout.write((char *) pszDataBuf, nDataLen);
    fout.close();

    delete[] pszDataBuf;

    std::cout << "save() suc" << std::endl;
    return true;
}


bool FlatIndex::load(string strFileName) {
    release();

    ifstream fin;
    fin.open(strFileName.c_str(), ios::binary);
    if (!fin.is_open()) {
        cerr << "failed to open the file:" << strFileName << " for reading." << endl;
        return false;
    }

    fin.seekg(0, ios::end);
    int nDataLen = fin.tellg();
    unsigned char *pszDataBuf = new(std::nothrow) unsigned char[nDataLen];
    if (pszDataBuf == NULL) {
        spdlog::info("load, pszDataBuf 分配内存失败! ! ");
        return false;
    }
    fin.seekg(0, ios::beg);
    fin.read((char *) pszDataBuf, nDataLen);
    fin.close();

    if (!buffer2index(pszDataBuf, nDataLen)) {
        return false;
    }
    delete[] pszDataBuf;

    m_nLoadTimes += 1;

    std::cout << "load() suc" << std::endl;
    return true;
}


int FlatIndex::getNxInsertNum() {
    return m_nInsertNum;
}


int FlatIndex::encrypt_save(string strFileName) {
    int nDataLen = 0;
    nDataLen += sizeof(int) * 10;
    nDataLen += sizeof(float) * m_nFeaLen * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(int) * m_nListLen;
    nDataLen += sizeof(char) * m_OssLength * m_nListLen;
    spdlog::info("encrypt_save------nDataLen::{d}", nDataLen);

    unsigned char *pszDataBuf = new(std::nothrow) unsigned char[nDataLen];
    if (pszDataBuf == NULL) {
        spdlog::info("encrypt_save, pszDataBuf 分配内存失败! ! ");
        return -1;
    }
    save_index++;
    int now_index = save_index % 3;
    strFileName = strFileName + "_" + to_string(now_index);

    spdlog::info("encrypt_save-4:  filename %s {d} {d} ", strFileName.c_str(), save_index,
         now_index);
    if (index2buffer(pszDataBuf, nDataLen) != true) {
        return -2;
    }

    if (m_encryptor.encrypt(pszDataBuf, nDataLen) != true) {
        return -1;
    }

    ofstream fout;
    fout.open(strFileName.c_str(), ios::binary);
    if (!fout.is_open()) {
        spdlog::info("encrypt_save------open error: %s", strFileName.c_str());
        return -3;
    }
    fout.write((char *) pszDataBuf, nDataLen);
    fout.close();

    delete[] pszDataBuf;

    spdlog::info("encrypt_save() suc");
    spdlog::info("encrypt_save------m_nListLen::{d}", m_nListLen);
    spdlog::info("encrypt_save------m_nMaxClassID:{d}", m_nMaxClassID);
    spdlog::info("encrypt_save------m_nInsertNum:{d}", m_nInsertNum);
    spdlog::info("encrypt_save------m_nLoadTimes:{d}", m_nLoadTimes);
    spdlog::info("encrypt_save------m_nSearchTimes:{d}", m_nSearchTimes);
    spdlog::info("encrypt_save------m_nRegistTimes:{d}", m_nRegistTimes);

    return now_index;
}


bool FlatIndex::decrypt_load(string strFileName) {
    spdlog::info("buffer2index, 打开索引文件! ! {}", strFileName);
    release();

    ifstream fin;
    fin.open(strFileName.c_str(), ios::binary);
    if (!fin.is_open()) {
        spdlog::info("buffer2index, 打开索引文件失败! ! {}", strFileName);
        return false;
    }

    spdlog::info("buffer2index, 开始执行 decrypt_load");
    fin.seekg(0, ios::end);
    int nDataLen = fin.tellg();
    unsigned char *pszDataBuf = new(std::nothrow) unsigned char[nDataLen];
    if (pszDataBuf == NULL) {
        spdlog::info("decrypt_load, pszDataBuf 分配内存失败! ! ");
        return false;
    }
    fin.seekg(0, ios::beg);
    fin.read((char *) pszDataBuf, nDataLen);
    fin.close();

    if (!m_encryptor.decrypt(pszDataBuf, nDataLen)) {
        spdlog::info("buffer2index, 解密失败! ! ");
        return false;
    }
    spdlog::info("buffer2index:nDataLen={}", nDataLen);
    if (!buffer2index(pszDataBuf, nDataLen)) {
        spdlog::info("buffer2index, 缓存转索引失败! ! ");
        return false;
    }
    delete[] pszDataBuf;

    m_nLoadTimes += 1;
    spdlog::info("load nx suc");

    return true;
}


bool FlatIndex::saveForbiddens(int* forbiddenIDs, int forbiddenSize) {
    spdlog::info("setForbiddenSaleClsID, 分配/重新分配内存开始");
    m_pForbiddens = new(std::nothrow) bool[m_nListLen];
    if (m_pForbiddens == NULL) {
        spdlog::info("setForbiddenSaleClsID, 分配内存失败");
        return false;
    }
    spdlog::info("setForbiddenSaleClsID, 分配内存成功! {d} ", sizeof(bool));

    memset(m_pForbiddens, 0, sizeof(bool) * m_nListLen);
    spdlog::info("setForbiddenSaleClsID, 禁售类别数={d} ", forbiddenSize);
    for(int i=0; i<min(m_nInsertNum, m_nListLen); i++) {
        for (int j=0; j<forbiddenSize; j++) {
            if (m_pnClsIDMap[i] == forbiddenIDs[j]) {
                m_pForbiddens[i] = true;
                break;
            }
        }
    }
    spdlog::info("setForbiddenSaleClsID forbiddens set Done!");
    return true;
}



