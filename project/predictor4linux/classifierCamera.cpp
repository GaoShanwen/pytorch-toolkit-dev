// #include <omp.h>
// #include "utils.h"
// #include "stdafx.h"

#include <dlfcn.h>
#include "classifierCamera.h"
#include "spdlog/spdlog.h"
#include <spdlog/cfg/env.h>  // support for loading levels from the environment  variable
#include <spdlog/fmt/ostr.h> // support for user defined types

using namespace spdlog;
using namespace std;

ClassifierCamera::ClassifierCamera()
{
}


ClassifierCamera::~ClassifierCamera()
{
}

string ClassifierCamera::get_model_type()
{
	return "camera";
}

bool ClassifierCamera::init(string com_port, int threads)
{
	hDllInst = dlopen("libRvPredC.so", RTLD_LAZY);
	if (!hDllInst)
	{
		spdlog::error("load libRvPredC.so faild !!!");
		return false;
	}

	// 获取函数指针
    rv_init = (RvInit)dlsym(hDllInst, "RvInit");
    if (!rv_init) {
        spdlog::error("not found RvInit method in RvPredC64.dll");
        dlclose(hDllInst);
        return false;
    }

    rv_close = (RvClose)dlsym(hDllInst, "RvClose");
	if (!rv_close) return false;

    rv_pred = (RvPred)dlsym(hDllInst, "RvPred");
	if (!rv_pred) return false;

    rv_check = (RvCheck)dlsym(hDllInst, "RvCheck");
	if (!rv_check) return false;

    rv_cap = (RvCapture)dlsym(hDllInst, "RvCapture");
	if (!rv_cap) return false;

    rv_save_pic = (savePic)dlsym(hDllInst, "savePic");
	if (!rv_save_pic) return false;

	const char* port = com_port.c_str();
	int exists_camera = rv_init((char*)port);
	spdlog::info("camera init :{}", exists_camera);
	return exists_camera == 0 ? true : false;
}

int ClassifierCamera::release()
{
	if (!hDllInst)
	{
		return -1;
	}

	if (!rv_close)
	{
		return -2;
	}

	dlclose(hDllInst);
	return 0;
}
int ClassifierCamera::predict(float input_data_arr[], int some_size)
{
	spdlog::info("rv_pred start...");
	if (!rv_pred)
	{
		spdlog::error("rv_pred failed!");
		return -1;
	}
	return 0;
}

bool ClassifierCamera::get_prob(float * pfProb, int nClassNum)
{
	return true;
}


int split(const string& s, vector<float>& sv, const char flag = ' ')
{
	sv.clear();
	istringstream iss(s);
	string temp;

	while (getline(iss, temp, flag))
	{
		sv.push_back(stof(temp));
	}
	return sv.size();
}

//string keyword = "[";
int ClassifierCamera::get_fea_dim()
{
	return this->nDim;
}

bool ClassifierCamera::get_fea(float *pfFea, int nFeaDim)
{
	spdlog::info("RvPred malloc buffer init...");
	char *predbuf = NULL;
	int predlen = 1920 * 1080 + 1;
	predbuf = (char*)malloc(predlen);
	if (predbuf == NULL)
	{
		spdlog::error("RvPred malloc buffer faild!");
		return -1;
	}
	memset(predbuf, 0x00, predlen);
	spdlog::info("RvPred malloc buffer init3...");
	int ret = rv_pred(predbuf, predlen);
	spdlog::info("RvPred malloc buffer init4...");

	string sFeature(predbuf);

	vector<float> v;
	nFeaDim = split(sFeature, v, ',');
	this->nDim = nFeaDim;
	pfFea = reinterpret_cast<float*>(v.data());
	
	free(predbuf);
	return true;
}


bool ClassifierCamera::get_res(int * pnResID, float * pfResScore, int nTopK)
{
	return true;
}
bool ClassifierCamera::load(string strFileName)
{
	return true;
}
bool ClassifierCamera::decrypt_load(string com_port)
{
	return init(com_port,1);
}
bool ClassifierCamera::decrypt_load(string com_port, string model_filename, string params_filename)
{
	return init(com_port, 1);
}