#pragma once
#include "IClassifier.h"

using namespace std;

class ClassifierFactory {
public:
    ClassifierFactory();
    ~ClassifierFactory();

    // static std::shared_ptr<IClassifier> create(const int classifier_type);
	static IClassifier* create(const string& type = "");

};
