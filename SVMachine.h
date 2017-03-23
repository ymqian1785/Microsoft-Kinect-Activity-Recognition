#pragma once
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>
#include "stdafx.h"
#include <strsafe.h>
#include <math.h>
#include <limits>
#include <Wincodec.h>
#include "iostream"
#include <windows.h>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <string>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
class SVMachine
{
public:
	SVMachine();
	~SVMachine();
	void Run(std::vector<float> l_traindata);
protected:
	int     Predict(cv::Mat l_traindata, cv::Ptr<cv::ml::SVM> l_svm);
private:
	bool	keyWasPressed(int vkcode);
	void    Train();

};


