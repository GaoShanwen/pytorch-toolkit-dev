#ifndef ICLASSIFIER_H
#define ICLASSIFIER_H


#include <string>

using namespace std;

class IClassifier
{
public:
//     virtual ~IClassifier() = default;
//     virtual bool init(string strModelFile) = 0;
//     virtual int predict(Mat img) = 0;
//     virtual int release() = 0;

//     virtual bool get_prob(float * pfProb, int nClassNum) = 0;
//     virtual int get_fea_dim() = 0;
//     virtual bool get_fea(float * pfFea, int nFeaDim) = 0;
// //    virtual bool get_res(int * pnResID, float * pfResScore, int nTopK) = 0;

// //    virtual bool load(string strFileName) = 0;
//     virtual bool decrypt_load(string strFileName) = 0;

//    virtual bool decrypt_load(string model_dir_c, string model_filename, string params_filename) = 0;

	virtual bool init(string strModelFile, int threads) = 0;
	virtual int predict(float input_data_arr[], int some_size) = 0;
	virtual int release() = 0;

	virtual bool get_prob(float * pfProb, int nClassNum) = 0;
	virtual int get_fea_dim() = 0;
	virtual bool get_fea(float * pfFea, int nFeaDim) = 0;
	virtual bool get_res(int * pnResID, float * pfResScore, int nTopK) = 0;

	virtual bool load(string strFileName) = 0;
	virtual bool decrypt_load(string strFileName) = 0;
	virtual bool decrypt_load(string model_dir_c, string model_filename, string params_filename) = 0;

	virtual string get_model_type() = 0;
};

#endif // ICLASSIFIER_H