#pragma comment(lib, "user32.lib")
#include "SVMachine.h"
#define KEYDOWN(vkcode) (GetAsyncKeyState(vkcode) & 0x8000 ? true : false)
#define VK_1 0x31
#define VK_2 0x32
#define VK_3 0x33
#define TRAIN_DATA "X.xml"
#define LABEL_DATA "Y.xml"
#define COUNT_DATA "count.xml"

 SVMachine::SVMachine()
{
}


SVMachine::~SVMachine()
{
} 

void SVMachine::Run(std::vector<float> l_traindata)
{
	//std::cout << l_traindata.size() << std::endl;
	char menutext[] = "----------------------------------------------------------------------------\n"
		"*** SVM Menu (Press # to select option) *** \n"
		"----------------------------------------------------------------------------\n"
		"(1) Go back to main menu.\n"
		"(2) Train SVM\n";
	std::cout << menutext;
	if (KEYDOWN(VK_1))
		return;
	while (!keyWasPressed(VK_1))
		{		
			if (KEYDOWN(VK_2)) {
				Sleep(100);
				system("cls");
				Train();
				system("cls");
				std::cout << menutext;
				Sleep(100);
			}
			/*else if (KEYDOWN(VK_3)) {
				Sleep(100);
				system("cls");
				//int activity = Predict(l_traindata);
				Predict();
			 //   std::cout << "The activity is : " << activity << std::endl;
			//	std::cout << "Press 1 to exit" << std::endl;
			}*/
		}
	system("cls");
	std::cout << menutext;
	Sleep(100);
	}

void SVMachine::Train(){
	char video_count[50];
	cv::Mat temp;
	cv::Mat trainData ;
	cv::Mat labels;
	// Find the count
	CvFileStorage*  fs = cvOpenFileStorage(COUNT_DATA, 0, CV_STORAGE_READ);
	CvFileNode* fn = cvGetFileNodeByName(fs, 0, "count");
	//CvFileNode*  fn1 = cvGetFileNode(fs, fn, cvGetHashedKey(fs, "rows", -1, 0), 0);
	int count_size = cvReadIntByName(fs, fn, "data");
	cvReleaseFileStorage(&fs);
	cv::FileStorage read_2a(TRAIN_DATA, cv::FileStorage::READ); //FileStorage::READ  
	for (int i = 1; i <= count_size; i++)
	{
		sprintf_s(video_count, "video%d", i);
		read(read_2a[video_count], temp);
		trainData.push_back(temp);
	}
	read_2a.release();
	//read label xml file
	cv::FileStorage read_2b(LABEL_DATA, cv::FileStorage::READ);
	//read data into Mat  
	read(read_2b["Labels"], labels);
	read_2b.release();
	/*int label[2] = { 1, 1 };
	cv::Mat labels(2, 1, CV_32SC1, label);*/
	// Set up SVM's parameters
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::NU_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 1000, 1e-6));
	svm->setNu(0.2);
	// Train the SVM with given parameters
	cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, labels);
	svm->train(td);
	svm->save("SVM.xml");
}

int SVMachine::Predict(cv::Mat l_traindata, cv::Ptr<cv::ml::SVM> l_svm){
	//  void SVMachine::Predict(){
	int response = l_svm->predict(l_traindata);
	return response;

	/* char video_count[50];
	cv::Mat temp;
	cv::Mat temp2 = cv::Mat(16, 159200, CV_32FC1);
	std::cout << temp2.size();
	cv::Mat trainData;
	cv::Mat labels;
	// Find the count
	CvFileStorage*  fs = cvOpenFileStorage(COUNT_DATA, 0, CV_STORAGE_READ);
	CvFileNode* fn = cvGetFileNodeByName(fs, 0, "count");
	//CvFileNode*  fn1 = cvGetFileNode(fs, fn, cvGetHashedKey(fs, "rows", -1, 0), 0);
	int count_size = cvReadIntByName(fs, fn, "data");
	cvReleaseFileStorage(&fs);
	cv::FileStorage read_2a(TRAIN_DATA, cv::FileStorage::READ); //FileStorage::READ  
	for (int i = 1; i <= count_size; i++)
	{
		sprintf_s(video_count, "video%d", i);
		read(read_2a[video_count], temp);
		trainData.push_back(temp);
	}
	read_2a.release();
	
	// Load the SVM from memory
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load<cv::ml::SVM>("SVM.xml");
	int response;
	for (int i = 0; i < 16; i++)
	{
	temp2.push_back(trainData.row(i));
	response = svm->predict(temp2.row(i));
	std::cout << response << std::endl; 
	} */
	/*
	int col = ll_traindata.size();
	std::cout << ll_traindata.size();
	//convert feature vect to Mat form
	cv::Mat featMat = cv::Mat(1, col, CV_32FC1);
	memcpy(featMat.data, ll_traindata.data(), ll_traindata.size()*sizeof(float));
	// Load the SVM from memory
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load<cv::ml::SVM>("SVM.xml");
	int response = svm->predict(featMat);
	return response;  */
	//return 1;
}

/// <summary>
/// Checks if a specific key has been pressed
/// </summary>
/// <param name="vkcode">keyboard key code</param>
bool SVMachine::keyWasPressed(int vkcode)
{
	short ks;
	short LSB;
	short MSB;

	ks = GetAsyncKeyState(vkcode);
	LSB = ks & 1;
	MSB = ks > 0;
	if (LSB || MSB) return true;
	return false;
}
