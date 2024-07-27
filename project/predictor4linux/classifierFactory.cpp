// #include <memory>
// #include "smart_predictor.h"
// #include "stdafx.h"
#include "classifierFactory.h"
#include "classifierCamera.h"

#include "spdlog/spdlog.h"
#include <spdlog/cfg/env.h>  // support for loading levels from the environment  variable
#include <spdlog/fmt/ostr.h> // support for user defined types

using namespace spdlog;
using namespace std;

ClassifierFactory::ClassifierFactory()
{
}


ClassifierFactory::~ClassifierFactory()
{
}

bool startsWithIgnoreCase(const string str, const char* prefix) {
	return str.compare(0, strlen(prefix), prefix) == 0;
}

// std::shared_ptr<IClassifier> ClassifierFactory::create(const int classifier_type) {
//     if (classifier_type == SMARTPREDICTOR_CAMERA) {
//         spdlog::info("create camera model {}", "SMARTPREDICTOR_CAMERA");
//         return std::make_shared<ClassifierCamera>();
//     }
//     if (classifier_type == SMARTPREDICTOR_PDCPU) {
//         spdlog::info("create Paddle model {}", "no support yet!");
//         return nullptr;
//         // return std::make_shared<ClassifierPDCpu>();
//     }
//     if (classifier_type == SMARTPREDICTOR_RKNPU) {
//         spdlog::info("create Rockchip model {}", "no support yet!");
//         return nullptr;
//         // return std::make_shared<ClassifierRKNpu>();
//     }

//     return nullptr;
// };
IClassifier* ClassifierFactory::create(const string& type)
{
	spdlog::info("Creating Classifier of type: {}", type);
	bool flag = startsWithIgnoreCase(type, "/dev/tty");
	spdlog::info("Creating Classifier flag0 = {}", flag);
	flag = type.empty()==false;
	spdlog::info("Creating Classifier flag1 = {}", flag);
	if (type.empty()==false && (startsWithIgnoreCase(type, "com") || startsWithIgnoreCase(type, "/dev/tty")))
	{
		spdlog::info("camera classifer init...");
		return new ClassifierCamera();
	}

	spdlog::error("classifer init failed!");
	// IClassifier* classifier = new IClassifier();
	return NULL;
}
