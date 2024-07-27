#pragma once
#include "IClassifier.h"


using namespace std;

class ClassifierCamera : public IClassifier {

private:
    string read_file(string strFileName);
    static bool vCompare(const pair<int, float> p1, const pair<int, float> p2);

public:
    ClassifierCamera();
    ~ClassifierCamera();
    // bool init(string strModelFile);

    // int release();
    // int predict(Mat img);

    // bool get_prob(float * pfProb, int nClassNum);
    // int get_fea_dim();
    // bool get_fea(float * pfFea, int nFeaDim);
    // bool decrypt_load(string strFileName);
	bool init(string strModelFile, int threads);
	int predict(float input_data_arr[], int some_size);

	bool get_prob(float * pfProb, int nClassNum);
	int get_fea_dim();
	bool get_fea(float * pfFea, int nFeaDim);
	bool get_res(int * pnResID, float * pfResScore, int nTopK);
	bool load(string strFileName);
	bool decrypt_load(string strFileName);
	bool decrypt_load(string model_dir_c, string model_filename, string params_filename);

	int release();
	string get_model_type();
private:
    // int m_nFeaDim;
    // float * m_pfFea;
	int nDim=128;

	void *hDllInst = NULL; // Linux下使用void*作为动态库句柄的类型

	typedef int (*RvInit)(char *commname);
	RvInit rv_init;

	typedef int (*RvClose)();
	RvClose rv_close;

	typedef int (*RvCheck)();
	RvCheck rv_check;

	typedef int (*RvPred)(char *retbuf, int retmax);
	RvPred rv_pred;

	typedef int (*RvCapture)(char *retbuf, int retmax);
	RvCapture rv_cap;

	typedef int (*savePic)(char *picname, char *picdata, int piclen);
	savePic rv_save_pic;
};