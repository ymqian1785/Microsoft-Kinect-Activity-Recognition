//------------------------------------------------------------------------------
// Callib.cpp
// 
// University of Florida
// Francesco Pittaluga
// f.pittaluga@ufl.edu
//------------------------------------------------------------------------------

#include "Callib.h"

#define KEYDOWN(vkcode) (GetAsyncKeyState(vkcode) & 0x8000 ? true : false)
#define VK_1 0x31
#define VK_2 0x32
#define VK_3 0x33

using namespace std;
using namespace cv;


/// <summary>
/// Constructor
/// </summary>
Callib::Callib(std::string mainWinName, std::string sideWinName, std::string dispWinName, std::string dispCallibFileName, std::string infraredCallibFileName) :
m_pKinectSensor(NULL),
m_pMultiSourceFrameReader(NULL),
m_pInfraredFrameReader(NULL),
m_pCoordinateMapper(NULL),
m_pColorFrameReader(NULL),
m_pInfraredRGBX(NULL),
m_pColorRGBX(NULL),
cDisplayCallibFileName(dispCallibFileName),
cInfraredCallibFileName(infraredCallibFileName),
cMainWinName(mainWinName),
cDisplayWinName(dispWinName),
cSecondWinName(sideWinName)
{
	// Create heap storage for color/infrared pixel data in RGBX format
	m_pInfraredRGBX = new RGBQUAD[cInfraredWidth * cInfraredHeight];
	m_pColorRGBX = new RGBQUAD[cColorWidth * cColorHeight];
}

/// <summary>
/// Destructor
/// </summary>
Callib::~Callib()
{
	// clean up Direct2D renderer
	if (m_pInfraredRGBX)
	{
		delete[] m_pInfraredRGBX;
		m_pInfraredRGBX = NULL;
	}
	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	// done with frame readers
	SafeRelease(m_pMultiSourceFrameReader);
	SafeRelease(m_pInfraredFrameReader);
	SafeRelease(m_pColorFrameReader);

	// done the Kinect Sensor
	SafeRelease(m_pKinectSensor);
}


void Callib::Run()
{

	// Initialize kinect 
	InitializeDefaultKinectSensors();

	bool isOutputWindowInit = false;


	string menuText = "----------------------------------------------------------------------------\n"
		 "*** Callibration Menu (Press # to select option) *** \n"
		 "----------------------------------------------------------------------------\n"
		 "(1) Go Back to Main Menu.\n"
		 "(2) Infrared Callibration.\n"
		 "(3) Display Callibration.\n\n";

	std::cout << menuText;

	// Main callibration loop
	while (true)
	{
		if (KEYDOWN(VK_ESCAPE) || KEYDOWN(VK_1)) {
			return;
		}
		else if (KEYDOWN(VK_2)) {
			Sleep(100);
			system("cls");
			RunInfraredCallibration();
			Sleep(100);
			system("cls");
			std::cout << menuText;
		}
		else if (KEYDOWN(VK_3)) {
			system("cls");
			RunDisplayCallibration();
			Sleep(100);
			system("cls");
			std::cout << menuText;
		}
	}


	// Close Kinect
	if (m_pKinectSensor && !m_WasKinectOpenWhenInitialized)
	{
		m_pKinectSensor->Close();
	}
		
}


/// <summary>
/// Display/Beamsplitter/Color Camera Callibration
/// </summary>
/// <summary>
/// Main callibration function
/// </summary>
void Callib::RunDisplayCallibration()
{
	cout << "----------------------------------------------------------------------------\n"
		<< "*** Display Callibration (Press ESC to go back Callibration Menu) *** \n"
		<< "----------------------------------------------------------------------------\n";

	// Init display window
	string dispWinName = cDisplayWinName;
	//Mat chessboardImg = imread("chessboard.JPG");
	Mat chessboardImg = Mat(cColorHeight, cColorWidth, CV_8UC3);
	resize(imread("chessboard.JPG"), chessboardImg, chessboardImg.size(), 0, 0, INTER_LINEAR);
	imshow(dispWinName, chessboardImg); waitKey(33);

	// Init main window
	string colorWinName = cMainWinName;

	//////////////////////////////
	// Get valid infrared mapping
	Mat infraredRoiMask;
	Mat mappedInfraredMask = Mat(cColorHeight, cColorWidth, CV_8UC1, Scalar(1));
	cv::FileStorage fs1(cInfraredCallibFileName, cv::FileStorage::READ);
	read(fs1["MASK"], infraredRoiMask);
	fs1.release();

	DepthSpacePoint* pDepthCoordinates = new DepthSpacePoint[cColorWidth * cColorHeight];
	UINT16 pDepthBuffer[cDepthHeight * cDepthWidth];

	for (int i = 0; i < (cDepthWidth * cDepthHeight); i++)
	{
		if (infraredRoiMask.data[i] == 1)
			pDepthBuffer[i] = 1000000;
		else
			pDepthBuffer[i] = 0;
	}

	HRESULT hr = m_pCoordinateMapper->MapColorFrameToDepthSpace(cDepthWidth * cDepthHeight, (UINT16*)pDepthBuffer, cColorWidth * cColorHeight, pDepthCoordinates);

	if (SUCCEEDED(hr))
	{
		cv::Vec4b zeroVec = cv::Vec4b((byte)0);

		// loop over output pixels
		for (int colorIndex = 0; colorIndex < (cColorWidth * cColorHeight); colorIndex++)
		{
			DepthSpacePoint p = pDepthCoordinates[colorIndex];
			int depthX = static_cast<int>(p.X + 0.5f);
			int depthY = static_cast<int>(p.Y + 0.5f);

			if (!((depthX >= 0 && depthX < cDepthWidth) && (depthY >= 0 && depthY < cDepthHeight)))
			{
				int y = one2twoD(colorIndex, cColorWidth, 'r');
				int x = one2twoD(colorIndex, cColorWidth, 'c');
				mappedInfraredMask.at<byte>(y, x) = 0;
			}
		}
	}


	////////////////////////////////////////////
	// Get Display Region of Interest From User

	Mat colorFrame;
	Mat dispFrame;
	Mat iframe;
	Mat rawColorFrame;

	// Cout instructions
	cout << "Step 1: Select Region of Interest (ROI)" << endl;
	cout << "     a) Click on top-left corner of ROI" << endl;
	cout << "     b) Click on bottom-right corner of ROI" << endl;

	// Set mouse callback for new window
	vector<Point> dispClickedPoints;
	cv::setMouseCallback(dispWinName, onMouse, (void*)&dispClickedPoints); //set mouse callback for new window

	// Get valid region corner points from user
	dispFrame = Mat(cColorHeight, cColorWidth, CV_8UC3, Scalar(0, 0, 255));
	while (true)
	{
		if (keyWasPressed(VK_ESCAPE)) { //user pressed ESC-key
			cout << endl;
			return;
		}
		imshow(dispWinName, dispFrame);

		iframe = GetInfraredBufferAsMat();
		imshow(cSecondWinName, iframe);

		rawColorFrame = GetColorBufferAsMat();
		rawColorFrame.copyTo(colorFrame, mappedInfraredMask);
		imshow(colorWinName, colorFrame); waitKey(33);

		if (dispClickedPoints.size() > 1)
		{
			break;
		}


	}

	// Display checkerboard in selected roi
	Mat tempChessboardImg = Mat(dispClickedPoints[1].y - dispClickedPoints[0].y, dispClickedPoints[1].x - dispClickedPoints[0].x, CV_8UC3);
	resize(chessboardImg, tempChessboardImg, tempChessboardImg.size(), 0, 0, INTER_LINEAR);
	Rect displayRoi = Rect(dispClickedPoints[0].x, dispClickedPoints[0].y, tempChessboardImg.cols, tempChessboardImg.rows);
	Mat smallShiftedChessboardImg = Mat(cColorHeight, cColorWidth, CV_8UC3);
	tempChessboardImg.copyTo(smallShiftedChessboardImg(displayRoi));
	imshow(dispWinName, smallShiftedChessboardImg*0.6); waitKey(33);


	////////////////////////////////////////////////
	// Get Homography by Capturing Checkerboard Img


	// cout instructions
	cout << "Step 1: Press C to capture image of chessboard" << endl;
	
	//Capture img of chessboard with color camera and detect board corners
	vector<Point2f> frameCornerPoints;
	Size boardSize = cvSize(8, 6);
	bool patternFound;
	while (true)
	{
		if (keyWasPressed(VK_ESCAPE)) { //user pressed ESC-key
			cout << endl;
			return;
		}
		if (keyWasPressed(cVK_C)) //user pressed C-key 
		{
			patternFound = findChessboardCorners(rawColorFrame,  //Get chessboard corners from frame
				boardSize, frameCornerPoints, CALIB_CB_ADAPTIVE_THRESH +
				CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
	
			if (patternFound)
			{
				break;
			}
			else
			{
				cout << "No corners found. Please try again." << endl << endl;
				cout << "Step 1: Press C to capture image of chessboard" << endl;
			}
		}
	
		iframe = GetInfraredBufferAsMat();
		imshow(cSecondWinName, iframe);
	
		rawColorFrame = GetColorBufferAsMat();
		rawColorFrame.copyTo(colorFrame, mappedInfraredMask);
		imshow(colorWinName, colorFrame);
		waitKey(1);
	}
	
	// Get display chessboard corner points 
	vector<Point2f> displayCornerPoints;
	findChessboardCorners(smallShiftedChessboardImg, boardSize, displayCornerPoints);
	
	// Compute homography b/w display cornerpoints and frame cornerpoints
	Mat H = findHomography(frameCornerPoints, displayCornerPoints);
	

	/////////////////
	// Get Color ROI

	// Cout instructions
	cout << "Step 2: Select Region of Interest (ROI)" << endl;
	cout << "     a) Click on top-left corner of ROI" << endl;
	cout << "     b) Click on bottom-right corner of ROI" << endl;
	
	// Set mouse callback for new window
	vector<Point> clickedPoints;
	cv::setMouseCallback(colorWinName, onMouse, (void*)&clickedPoints); //set mouse callback for new window
	
	// Get valid region corner points from user
	drawChessboardCorners(rawColorFrame, boardSize, Mat(frameCornerPoints), patternFound);
	colorFrame = dispFrame * 0;
	rawColorFrame.copyTo(colorFrame, mappedInfraredMask);
	while (true)
	{
		if (keyWasPressed(VK_ESCAPE)) { //user pressed ESC-key
			cout << endl;
			return;
		}
		imshow(colorWinName, colorFrame);
		if (clickedPoints.size() > 1)
		{
			break;
		}
		waitKey(33);
	}
	
	// Creat mask from corner points
	Mat finalRoiMask = Mat(cColorHeight, cColorWidth, CV_8UC1, Scalar(0));
	finalRoiMask(Rect(clickedPoints[0], clickedPoints[1])) = 1;
	
	// Embedd selected roi in display image
	Mat callibratedDispImg = chessboardImg.clone() * 0;
	vector<Point2f> origRect;
	vector<Point2f> warpedRect;
	origRect.push_back(Point2f(clickedPoints[0].x, clickedPoints[0].y));
	origRect.push_back(Point2f(clickedPoints[1].x, clickedPoints[1].y));
	
	Mat smallChessboardImg = Mat(clickedPoints[1].y - clickedPoints[0].y, clickedPoints[1].x - clickedPoints[0].x, CV_8UC3);
	resize(chessboardImg, smallChessboardImg, smallChessboardImg.size(), 0, 0, INTER_LINEAR);
	imshow(dispWinName, smallChessboardImg); waitKey(100);
	
	Mat chessboardImg2Display = Mat(cColorHeight, cColorWidth, CV_8UC3);
	Rect roi = Rect(clickedPoints[0].x, clickedPoints[0].y, smallChessboardImg.cols, smallChessboardImg.rows);
	smallChessboardImg.copyTo(chessboardImg2Display(roi));
	warpPerspective(chessboardImg2Display, callibratedDispImg, H, callibratedDispImg.size());
	imshow(dispWinName, callibratedDispImg); waitKey(33);
	
	// Show results
	cout << "Display callibration complete. Press ESC to return to Callibration Menu." << endl << endl;
	colorFrame = colorFrame * 0;
	while (!keyWasPressed(VK_ESCAPE))
	{
		rawColorFrame = GetColorBufferAsMat();
		rawColorFrame.copyTo(colorFrame, finalRoiMask);
		imshow(colorWinName, colorFrame);
		waitKey(33);
	}
	
	// Save callibration
	FileStorage fs(cDisplayCallibFileName, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "H" << H << "MASK" << finalRoiMask;
		fs.release();
	}
	else
	{
		cout << "Error: could not save the callibration parameters" << endl;
	}


	//////////////////////////////////////
	// Get Homography from chessboard Img

	//cout << "Step 2: Press C to capture image of chessboard" << endl;

	////Capture img of chessboard with color camera and detect board corners
	//while (true)
	//{
	//	if (keyWasPressed(VK_ESCAPE)) { //user pressed ESC-key
	//		cout << endl;
	//		return;
	//	}
	//	if (keyWasPressed(cVK_C)) //user pressed C-key 
	//	{
	//		patternFound = findChessboardCorners(colorFrame,  //Get chessboard corners from frame
	//			boardSize, frameCornerPoints, CALIB_CB_ADAPTIVE_THRESH +
	//			CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);

	//		if (patternFound)
	//		{
	//			break;
	//		}
	//		else
	//		{
	//			cout << "No corners found. Please try again." << endl << endl;
	//			cout << "Step 1: Press C to capture image of chessboard" << endl;
	//		}
	//	}

	//	iframe = GetInfraredBufferAsMat();
	//	imshow(cSecondWinName, iframe);

	//	colorFrame = GetColorBufferAsMat();
	//	colorFrame.copyTo(dispFrame, mappedInfraredMask);
	//	imshow(colorWinName, dispFrame);
	//	waitKey(1);
	//}

	//// Get display chessboard corner points 
	//vector<Point2f> displayCornerPoints;
	//findChessboardCorners(chessboardImg, boardSize, displayCornerPoints);

	//// Compute homography b/w display cornerpoints and frame cornerpoints
	//Mat H = findHomography(frameCornerPoints, displayCornerPoints);

	//// Embedd selected roi in display image
	//Mat callibratedDispImg = chessboardImg.clone() * 0;
	//vector<Point2f> origRect;
	//vector<Point2f> warpedRect;
	//origRect.push_back(Point2f(clickedPoints[0].x, clickedPoints[0].y));
	//origRect.push_back(Point2f(clickedPoints[1].x, clickedPoints[1].y));

	//drawChessboardCorners(colorFrame, boardSize, Mat(frameCornerPoints), patternFound);
	//warpPerspective(chessboardImg2Display, callibratedDispImg, H, callibratedDispImg.size());
	//imshow(dispWinName, callibratedDispImg); waitKey(33);

	//// Creat mask from corner points
	//Mat finalRoiMask = Mat(cColorHeight, cColorWidth, CV_8UC1, Scalar(0));
	//finalRoiMask(Rect(dispClickedPoints[0], dispClickedPoints[1])) = 1;

	//// Show results
	//cout << "Display callibration complete. Press ESC to return to Callibration Menu." << endl << endl;
	//dispFrame = dispFrame * 0;
	//while (!keyWasPressed(VK_ESCAPE))
	//{
	//	colorFrame = GetColorBufferAsMat();
	//	colorFrame.copyTo(dispFrame, finalRoiMask);
	//	imshow(colorWinName, dispFrame);
	//	waitKey(33);
	//}

	//// Save callibration
	//FileStorage fs(cDisplayCallibFileName, FileStorage::WRITE);
	//if (fs.isOpened())
	//{
	//	fs << "H" << H << "MASK" << finalRoiMask;
	//	fs.release();
	//}
	//else
	//{
	//	cout << "Error: could not save the callibration parameters" << endl;
	//}
}


/// <summary>
/// Infrared Lensless/Lenslet Callibration
/// </summary>
void Callib::RunInfraredCallibration()
{
	cout << "----------------------------------------------------------------------------\n"
		<< "*** Infrared Callibration (Press ESC to go back Callibration Menu) *** \n"
		<< "----------------------------------------------------------------------------\n";

	int numFeaturePoints = 4;
	vector<Point2f> lensletPoints;
	vector<Point2f> lenslessPoints;

	Mat frame[2];
	stringstream ss;
	string cams[2] = {"W/O Sleeve", "With Sleeve"};
	string winName = cMainWinName;// "Infrared Callibration";

	for (int j = 0; j < 2; j++)
	{
		// output instructions
		cout << "Step " << 2 * j + 1 << ": Press C to capture frame (" << cams[j] << ")." << endl;

		// get Frame
		while (true)
		{
			if (keyWasPressed(VK_ESCAPE)) {		  // user pressed ESC-key
				cout << endl;
				return;
			}

			if (keyWasPressed(cVK_C)){ break; }   // check for user input to capture frame

			frame[j] = GetInfraredBufferAsMat();
			imshow(winName, frame[j]);
			waitKey(1);
		}

		// output instructions
		cout << "Step " << 2 * j + 2 << ": Click on " << numFeaturePoints << " corner points in captured frame." << endl;

		// set mouse callback for new window
		vector<Point> points;
		cv::setMouseCallback(winName, onMouse, (void*)&points);

		// get corner points from user
		while (true)
		{
			if (keyWasPressed(VK_ESCAPE)) {       // user pressed ESC-key
				cout << endl;
				return;
			}
			imshow(winName, frame[j]);
			if (points.size() > numFeaturePoints - 1)
			{
				vector<Point2f> tempPoints;
				for (int k = 0; k < numFeaturePoints; k++)
				{
					tempPoints.push_back(points[k]);
				}
				if (j == 0) lenslessPoints = tempPoints;
				if (j == 1) lensletPoints  = tempPoints;
				break;
			}
			waitKey(33);
		}
	}

	//Compute Homography
	vector<Point2f> fixedPoints;
	Mat H = findHomography(lensletPoints, lenslessPoints);
	
	//Create mask
	Mat blackMat = Mat(cDepthHeight, cDepthWidth, CV_8UC1, Scalar(1));
	Mat mask = Mat(cDepthHeight, cDepthWidth, CV_8UC1);
	cv::warpPerspective(blackMat, mask, H, mask.size());

	//Show Results
	Mat im2disp;
	Mat warpedFrameLenslet = Mat(cDepthHeight, cDepthWidth, CV_8UC1);
	cv::warpPerspective(frame[1], warpedFrameLenslet, H, frame[1].size());
	addWeighted(frame[0], 0.5, warpedFrameLenslet, 0.5, 0.0, im2disp);
	cout << "Infrared callibration complete. Press ESC to return to Callibration Menu." << endl << endl;
	while (!keyWasPressed(VK_ESCAPE))
	{
		imshow(winName, im2disp);
		waitKey(1);
	}

	// Save callibration
	FileStorage fs(cInfraredCallibFileName, FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "H" << H << "MASK" << mask;
		fs.release();
	}
	else
	{
		cout << "Error: could not save the callibration parameters\n";
	}
}

/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT Callib::InitializeDefaultKinectSensors()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		hr = m_pKinectSensor->get_IsOpen(&m_WasKinectOpenWhenInitialized);
		if (SUCCEEDED(hr) && !m_WasKinectOpenWhenInitialized)
		{
			hr = m_pKinectSensor->Open();
		}

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
		}

		// get the infrared reader
		IInfraredFrameSource* pInfraredFrameSource = NULL;

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_InfraredFrameSource(&pInfraredFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pInfraredFrameSource->OpenReader(&m_pInfraredFrameReader);
		}

		SafeRelease(pInfraredFrameSource);


		// get the color reader
		IColorFrameSource* pColorFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}

		SafeRelease(pColorFrameSource);

	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		cout << "No ready Kinect found!";
		return E_FAIL;
	}

	return hr;
}


/// <summary>
/// Get color buffer
/// </summary>
void Callib::GetColorBuffer()
{
	if (!m_pColorFrameReader)
	{
		return;
	}

	IColorFrame* pColorFrame = NULL;

	HRESULT hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		UINT nBufferSize = 0;
		RGBQUAD *pBuffer = NULL;

		hr = pColorFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Width(&nWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&nHeight);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr))
		{
			if (imageFormat == ColorImageFormat_Bgra)
			{
				hr = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE**>(&pBuffer));
			}
			else if (m_pColorRGBX)
			{
				pBuffer = m_pColorRGBX;
				nBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(pBuffer), ColorImageFormat_Bgra);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		SafeRelease(pFrameDescription);
	}

	SafeRelease(pColorFrame);
}


/// <summary>
/// Get infrared buffer
/// </summary>
void Callib::GetInfraredBuffer()
{
	if (!m_pInfraredFrameReader)
	{
		return;
	}

	IInfraredFrame* pInfraredFrame = NULL;

	HRESULT hr = m_pInfraredFrameReader->AcquireLatestFrame(&pInfraredFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		UINT nBufferSize = 0;
		UINT16 *pBuffer = NULL;

		hr = pInfraredFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr)) hr = pInfraredFrame->get_FrameDescription(&pFrameDescription);
		if (SUCCEEDED(hr)) hr = pFrameDescription->get_Width(&nWidth);
		if (SUCCEEDED(hr)) hr = pFrameDescription->get_Height(&nHeight);
		if (SUCCEEDED(hr)) hr = pInfraredFrame->AccessUnderlyingBuffer(&nBufferSize, &pBuffer);

		if (SUCCEEDED(hr) && pBuffer && (nWidth == cInfraredWidth) && (nHeight == cInfraredHeight))
		{
			RGBQUAD* pDest = m_pInfraredRGBX;

			// end pixel is start + width*height - 1
			const UINT16* pBufferEnd = pBuffer + (nWidth * nHeight);
			
			/*UNT16 maxVal = 0;
			for (int i = 0; i < (nWidth * nHeight); i++)
			{
				if (*(pBuffer + 1) > maxVal)
				{
					maxVal = *(pBuffer + 1);
				}
			}

			cout << maxVal << endl;*/

			while (pBuffer < pBufferEnd)
			{
				// normalize the incoming infrared data (ushort) to a float ranging from 
				// [InfraredOutputValueMinimum, InfraredOutputValueMaximum] by
				// 1. dividing the incoming value by the source maximum value
				float intensityRatio = static_cast<float>(*pBuffer) / InfraredSourceValueMaximum;

				// 2. dividing by the (average scene value * standard deviations)
				intensityRatio /= InfraredSceneValueAverage * InfraredSceneStandardDeviations;

				// 3. limiting the value to InfraredOutputValueMaximum
				intensityRatio = min(InfraredOutputValueMaximum, intensityRatio);

				// 4. limiting the lower value InfraredOutputValueMinimym
				intensityRatio = max(InfraredOutputValueMinimum, intensityRatio);

				// 5. converting the normalized value to a byte and using the result
				// as the RGB components required by the image
				byte intensity = static_cast<byte>(intensityRatio * 255.0f);
				pDest->rgbRed = intensity;
				pDest->rgbGreen = intensity;
				pDest->rgbBlue = intensity;

				++pDest;
				++pBuffer;
			}
		}
		SafeRelease(pFrameDescription);
	}
	SafeRelease(pInfraredFrame);
}

/// <summary>
/// Initializes the console window
/// </summary>
/// <returns>indicates success or failure</returns>
void Callib::InitConsoleWindow()
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT outputMonitorRect = temp_vMonitorInfo->at(cOutputMonitor).rcWork;
	int outputMonitorWidth = outputMonitorRect.right - outputMonitorRect.left;
	int outputMonitorHeight = outputMonitorRect.bottom - outputMonitorRect.top;

	// Set console size/pos
	int consoleX1 = outputMonitorRect.left + outputMonitorWidth / 3 * 2;
	int consoleY1 = outputMonitorRect.top + outputMonitorHeight / 2;
	int consoleX2 = outputMonitorRect.left + outputMonitorWidth;
	int consoleY2 = outputMonitorRect.top + outputMonitorHeight + 10;
	RECT consoleWinRect = { consoleX1, consoleY1, consoleX2, consoleY2 };
	SetWindow2Rect(GetConsoleWindow(), consoleWinRect);

	// Set console title
	SetConsoleTitleA("Anonymous Kinect");

	delete(temp_vMonitorInfo);
}


/// <summary>
/// Initializes the console window
/// </summary>
/// <returns>indicates success or failure</returns>
void Callib::cvNamedDisplayWindow(const char * dispWinName)
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT displayMonitorRect = temp_vMonitorInfo->at(cDisplayMonitor).rcWork;

	// Init display window
	cvNamedWindow(dispWinName, CV_WINDOW_NORMAL);
	HWND displayWinHwnd = FindWindowA(0, dispWinName);
	SetWindowLongPtr(displayWinHwnd, GWL_STYLE, WS_SYSMENU | WS_POPUP | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE);
	SetWindow2Rect(displayWinHwnd, displayMonitorRect);

	delete(temp_vMonitorInfo);
}

/// <summary>
/// Initializes the display and output monitors
/// </summary>
/// <returns>indicates success or failure</returns>
void Callib::cvNamedMainWindow(const char * mainWinName)
{
	// Get monitor info
	vector<MONITORINFO> *temp_vMonitorInfo = new vector<MONITORINFO>;
	EnumDisplayMonitors(NULL, NULL, MonitorEnumProc, (LPARAM)temp_vMonitorInfo);
	RECT outputMonitorRect = temp_vMonitorInfo->at(cOutputMonitor).rcWork;
	int outputMonitorWidth = outputMonitorRect.right - outputMonitorRect.left;
	int outputMonitorHeight = outputMonitorRect.bottom - outputMonitorRect.top;

	// Set main window size/pos
	int mainWinX1 = outputMonitorRect.left;
	int mainWinY1 = outputMonitorRect.top;
	int mainWinX2 = mainWinX1 + outputMonitorWidth / 3 * 2;
	int mainWinY2 = mainWinY1 + outputMonitorHeight;
	RECT mainWinRect = { mainWinX1, mainWinY1, mainWinX2, mainWinY2 };
	cvNamedWindow(mainWinName, CV_WINDOW_NORMAL);
	HWND mainWinHwnd = FindWindowA(0, mainWinName);
	SetWindow2Rect(mainWinHwnd, mainWinRect);

	// clean up monitor info
	delete(temp_vMonitorInfo);
}


/// <summary>
/// Returns color buffer frame as a mat
/// </summary>
Mat Callib::GetColorBufferAsMat()
{
	Mat frame = Mat(cColorHeight, cColorWidth, CV_8UC3);
	GetColorBuffer();
	rgbquad2RGBMat(m_pColorRGBX, frame);
	return frame;
}


/// <summary>
/// Returns infrared buffer frame as a mat
/// </summary>
Mat Callib::GetInfraredBufferAsMat()
{
	Mat frame = Mat(cInfraredHeight, cInfraredWidth, CV_8UC3);
	GetInfraredBuffer();
	rgbquad2RGBMat(m_pInfraredRGBX, frame);
	return frame;
}


//*************************************************************//
//                *** Auxilliary Functions  ***                //
//*************************************************************//


/// <summary>
/// Checks if a specific key has been pressed
/// </summary>
/// <param name="vkcode">keyboard key code</param>
bool Callib::keyWasPressed(int vkcode)
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


/// <summary>
/// 1D array to 2D Mat
/// </summary>
void Callib::onMouse(int evt, int x, int y, int flags, void* param) {
	if (evt == CV_EVENT_LBUTTONDOWN) {
		std::vector<Point>* ptPtr = (vector<Point>*)param;
		ptPtr->push_back(Point(x, y));
	}
}

/// <summary>
/// Draws points onf frame
/// </summary>
void Callib::DrawPoints(vector<Point2f>* points, Mat frame, int radius, Scalar color) {
	for (int i = 0; i < (*points).size(); i++)
		circle(frame, cvPoint((*points)[i].x, (*points)[i].y), radius, color, -1, 8, 0);
}

/// <summary>
/// 1D array to 2D Mat
/// </summary>
/// <param name="oneD">One D array</param>
/// <param name="oneD">Two D Mat</param>
template <typename T, typename M>
void Callib::OneDByteArray2TwoDMat(T oneD, M twoD)
{
	int oneDIter = 0;
	for (int i = 0; i < twoD.rows; i++) {
		for (int j = 0; j < twoD.cols; j++) {
			twoD.at<byte>(i, j) = oneD[oneDIter];
			oneDIter++;
		}
	}
}

/// <summary>
/// converts rgb quad to mat
/// </summary>
void Callib::rgbquad2RGBMat(RGBQUAD* r, Mat m)
{
	RGBQUAD* pSrc = r;

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			m.at<Vec3b>(i, j)[0] = pSrc->rgbBlue;
			m.at<Vec3b>(i, j)[1] = pSrc->rgbGreen;
			m.at<Vec3b>(i, j)[2] = pSrc->rgbRed;
			++pSrc;
		}
	}
}



/// <summary>
/// Moves and resizes console window
/// </summary>
/// <returns>indicates success or failure</returns>
void Callib::SetWindow2Rect(HWND winHandle, RECT rect)
{
	int winWidth = rect.right - rect.left;
	int winHeight = rect.bottom - rect.top;;
	int winTopLeft_x = rect.left;
	int winTopLeft_y = rect.top;

	MoveWindow(winHandle, winTopLeft_x, winTopLeft_y, winWidth, winHeight, TRUE);

	int x = 0;
}


/// <summary>
/// Draws quadrilateral from 4 points
/// </summary>
/// <returns>indicates success or failure</returns>
void Callib::DrawQuadrilateral(Mat frame, Point2f topLeft, Point2f bottomLeft, Point2f topRight, Point2f BotttomRight, Scalar color, int thickness)
{
	line(frame, topLeft, topRight, color, thickness);
	line(frame, topLeft, bottomLeft, color, thickness);
	line(frame, BotttomRight, topRight, color, thickness);
	line(frame, BotttomRight, bottomLeft, color, thickness);
}

/// <summary>
/// Enumerate Monitors Callback
/// </summary>
BOOL CALLBACK Callib::MonitorEnumProc(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData)
{
	vector<MONITORINFO>* vMonitorInfo = (vector<MONITORINFO>*) dwData;
	MONITORINFO mi;
	mi.cbSize = sizeof(mi);
	GetMonitorInfo(hMonitor, &mi);
	vMonitorInfo->push_back(mi);
	return TRUE;
}


int Callib::two2oneD(int row, int col, int numCols)
{
	return row*numCols + col;
}

int Callib::one2twoD(int indx, int numCols, char rORc)
{
	if (rORc == 'r') return indx / numCols;
	else return indx%numCols;
}


















































///// <summary>
///// Display/Beamsplitter/Color Camera Callibration
///// </summary>
///// <summary>
///// Main callibration function
///// </summary>
//void Callib::RunDisplayCallibration()
//{
//	cout << "----------------------------------------------------------------------------\n"
//		<< "*** Display Callibration (Press ESC to go back Callibration Menu) *** \n"
//		<< "----------------------------------------------------------------------------\n";
//
//	// Init display window
//	string dispWinName = cDisplayWinName;
//	//Mat chessboardImg = imread("chessboard.JPG");
//	Mat chessboardImg = Mat(cColorHeight, cColorWidth, CV_8UC3);
//	resize(imread("chessboard.JPG"), chessboardImg, chessboardImg.size(), 0, 0, INTER_LINEAR);
//	imshow(dispWinName, chessboardImg); waitKey(33);
//
//	// Init main window
//	string colorWinName = cMainWinName;
//
//	// Get valid infrared mapping
//	Mat infraredRoiMask;
//	Mat mappedInfraredMask = Mat(cColorHeight, cColorWidth, CV_8UC1, Scalar(1));
//	cv::FileStorage fs1(cInfraredCallibFileName, cv::FileStorage::READ);
//	read(fs1["MASK"], infraredRoiMask);
//	fs1.release();
//
//	DepthSpacePoint* pDepthCoordinates = new DepthSpacePoint[cColorWidth * cColorHeight];
//	UINT16 pDepthBuffer[cDepthHeight * cDepthWidth];
//
//	for (int i = 0; i < (cDepthWidth * cDepthHeight); i++)
//	{
//		if (infraredRoiMask.data[i] == 1)
//			pDepthBuffer[i] = 1000000;
//		else
//			pDepthBuffer[i] = 0;
//	}
//
//	HRESULT hr = m_pCoordinateMapper->MapColorFrameToDepthSpace(cDepthWidth * cDepthHeight, (UINT16*)pDepthBuffer, cColorWidth * cColorHeight, pDepthCoordinates);
//
//	if (SUCCEEDED(hr))
//	{
//		cv::Vec4b zeroVec = cv::Vec4b((byte)0);
//
//		// loop over output pixels
//		for (int colorIndex = 0; colorIndex < (cColorWidth * cColorHeight); colorIndex++)
//		{
//			DepthSpacePoint p = pDepthCoordinates[colorIndex];
//			int depthX = static_cast<int>(p.X + 0.5f);
//			int depthY = static_cast<int>(p.Y + 0.5f);
//
//			if (!((depthX >= 0 && depthX < cDepthWidth) && (depthY >= 0 && depthY < cDepthHeight)))
//			{
//				int y = one2twoD(colorIndex, cColorWidth, 'r');
//				int x = one2twoD(colorIndex, cColorWidth, 'c');
//				mappedInfraredMask.at<byte>(y, x) = 0;
//			}
//		}
//	}
//
//	// cout instructions
//	cout << "Step 1: Press C to capture image of chessboard" << endl;
//
//	//Capture img of chessboard with color camera and detect board corners
//	Mat frame;
//	Mat dispFrame;
//	vector<Point2f> frameCornerPoints;
//	Size boardSize = cvSize(8, 6);
//	bool patternFound;
//	while (true)
//	{
//		if (keyWasPressed(VK_ESCAPE)) { //user pressed ESC-key
//			cout << endl;
//			return;
//		}
//		if (keyWasPressed(cVK_C)) //user pressed C-key 
//		{
//			patternFound = findChessboardCorners(frame,  //Get chessboard corners from frame
//				boardSize, frameCornerPoints, CALIB_CB_ADAPTIVE_THRESH +
//				CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
//
//			if (patternFound)
//			{
//				break;
//			}
//			else
//			{
//				cout << "No corners found. Please try again." << endl << endl;
//				cout << "Step 1: Press C to capture image of chessboard" << endl;
//			}
//		}
//
//		Mat iframe = GetInfraredBufferAsMat();
//		imshow(cSecondWinName, iframe);
//
//		frame = GetColorBufferAsMat();
//		frame.copyTo(dispFrame, mappedInfraredMask);
//		imshow(colorWinName, dispFrame);
//		waitKey(1);
//	}
//
//	// Get display chessboard corner points 
//	vector<Point2f> displayCornerPoints;
//	findChessboardCorners(chessboardImg, boardSize, displayCornerPoints);
//
//	// Compute homography b/w display cornerpoints and frame cornerpoints
//	Mat H = findHomography(frameCornerPoints, displayCornerPoints);
//
//	// Cout instructions
//	cout << "Step 2: Select Region of Interest (ROI)" << endl;
//	cout << "     a) Click on top-left corner of ROI" << endl;
//	cout << "     b) Click on bottom-right corner of ROI" << endl;
//
//	// Set mouse callback for new window
//	vector<Point> clickedPoints;
//	cv::setMouseCallback(colorWinName, onMouse, (void*)&clickedPoints); //set mouse callback for new window
//
//	// Get valid region corner points from user
//	drawChessboardCorners(frame, boardSize, Mat(frameCornerPoints), patternFound);
//	dispFrame = dispFrame * 0;
//	frame.copyTo(dispFrame, mappedInfraredMask);
//	while (true)
//	{
//		if (keyWasPressed(VK_ESCAPE)) { //user pressed ESC-key
//			cout << endl;
//			return;
//		}
//		imshow(colorWinName, dispFrame);
//		if (clickedPoints.size() > 1)
//		{
//			break;
//		}
//		waitKey(33);
//	}
//
//	// Creat mask from corner points
//	Mat finalRoiMask = Mat(cColorHeight, cColorWidth, CV_8UC1, Scalar(0));
//	finalRoiMask(Rect(clickedPoints[0], clickedPoints[1])) = 1;
//
//	// Embedd selected roi in display image
//	Mat callibratedDispImg = chessboardImg.clone() * 0;
//	vector<Point2f> origRect;
//	vector<Point2f> warpedRect;
//	origRect.push_back(Point2f(clickedPoints[0].x, clickedPoints[0].y));
//	origRect.push_back(Point2f(clickedPoints[1].x, clickedPoints[1].y));
//
//	Mat smallChessboardImg = Mat(clickedPoints[1].y - clickedPoints[0].y, clickedPoints[1].x - clickedPoints[0].x, CV_8UC3);
//	resize(chessboardImg, smallChessboardImg, smallChessboardImg.size(), 0, 0, INTER_LINEAR);
//	imshow(dispWinName, smallChessboardImg); waitKey(100);
//
//	Mat chessboardImg2Display = Mat(cColorHeight, cColorWidth, CV_8UC3);
//	Rect roi = Rect(clickedPoints[0].x, clickedPoints[0].y, smallChessboardImg.cols, smallChessboardImg.rows);
//	smallChessboardImg.copyTo(chessboardImg2Display(roi));
//	warpPerspective(chessboardImg2Display, callibratedDispImg, H, callibratedDispImg.size());
//	imshow(dispWinName, callibratedDispImg); waitKey(33);
//
//	// Show results
//	cout << "Display callibration complete. Press ESC to return to Callibration Menu." << endl << endl;
//	dispFrame = dispFrame * 0;
//	while (!keyWasPressed(VK_ESCAPE))
//	{
//		frame = GetColorBufferAsMat();
//		frame.copyTo(dispFrame, finalRoiMask);
//		imshow(colorWinName, dispFrame);
//		waitKey(33);
//	}
//
//	// Save callibration
//	FileStorage fs(cDisplayCallibFileName, FileStorage::WRITE);
//	if (fs.isOpened())
//	{
//		fs << "H" << H << "MASK" << finalRoiMask;
//		fs.release();
//	}
//	else
//	{
//		cout << "Error: could not save the callibration parameters" << endl;
//	}
//}