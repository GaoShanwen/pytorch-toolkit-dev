#include "encryptor.h"
#include <iostream>
#include <vector>
#include <string>

using namespace std;


class FlatIndex {
public:
    FlatIndex();

    ~FlatIndex();

    bool init(int nListLen, int nFeaLen, int nMaxClassID, int nInsertNum);

    void release();

    bool insert(const float *pfFea, int nDim, int nClassID, char *ossKey, int index, float limit);

    bool check_similarity(const float *pfFea, int nFeaDim, int setClassID);

    bool removeIndex(int index);

    bool getIndexByClassID(int nClassIDp, int pageno, int pageSize, int *indexID);

    bool searchKnn(const float *pfFea, int nDim, int *pnResID, float *pfResScore, int *indexID, float limit, int oriK, int finalK);

    bool train(float limit);

    string getOssKeyByIndex(int index);

    int getNxInsertNum();

    bool save(string strFileName);

    bool load(string strFileName);

    int encrypt_save(string strFileName);

    bool decrypt_load(string strFileName);

    bool saveForbiddens(int* forbiddens, int forbiddenSize);

private:
    static bool vCompare(const pair<int, float> p1, const pair<int, float> p2);

    static bool vCompareKnn(const pair<int, pair<int, float>> p1, const pair<int, pair<int, float>> p2);

    static bool vCompareEvery(const pair<int, pair<float, pair<int, float>*>> p1, const pair<int, pair<float, pair<int, float>*>> p2);

    static bool vCompareInt(const pair<int, int> p1, const pair<int, int> p2);

    inline bool normalize_l2(float *pfFea, int nDim);

    inline bool softmax(float *scores, int sSize);

    inline float similarity(const float *pfFea1, const float *pfFea2, int nDim);

    bool index2buffer(unsigned char *pszDataBuf, int nDataLen);

    bool buffer2index(unsigned char *pszDataBuf, int nDataLen);

private:
    int m_nListLen;
    int m_nFeaLen;
    int m_OssLength;
    int m_nMaxClassID;
    int m_nInsertNum;

    float *m_pfFeaMat;
    int *m_pnClsIDMap;
    int *m_pnClsIDMapSearchTimes;
    int *m_pnClsIDMapSelectTimes;
    int *m_pnClsIDMapSameTimes;
    int *m_pnClsIDMapReplaceTimes;
    bool *m_pForbiddens;
    char *m_pnClsIDOss;

    int m_nMaxLoadTimes;
    int m_nMaxSearchTimes;
    int m_nMaxRegistTimes;

    int m_nLoadTimes;
    int m_nSearchTimes;
    int m_nRegistTimes;

    int save_index;

    Encryptor m_encryptor;
};

