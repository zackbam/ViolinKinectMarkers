#include "app.h"
#include "util.h"
#include <vector>
#include <thread>
#include <chrono>
#include <inttypes.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

// Constructor
Kinect::Kinect()
{
	// Initialize
	initialize();
	smoothed_tilt = 0;
}

// Destructor
Kinect::~Kinect()
{
	// Finalize
	finalize();
}

typedef struct {
	int a=0, b=0;
} skline;
// Processing
void Kinect::run()
{
	// Main Loop
	while (true) {
		update();
		draw();
		infraredMat.convertTo(InfraFiltered, CV_32FC1);
		//float avg = 0;
		//for (int i = 0; i < infraredMat.cols; i++) {
		//	for (int j = 0; j < infraredMat.rows; j++) {
		//		//if (infraredMat.at<uint16_t>(cvPoint(i, j)) > avg)
		//			avg = avg + InfraFiltered.at<float>(j,i);
		//	}
		//}
		//cout << avg/ (float)infraredMat.cols*infraredMat.rows << endl;
		threshold(InfraFiltered, InfraFiltered, 28000, 255, CV_THRESH_BINARY);
		//infraredMat.convertTo(InfraFiltered, CV_8UC1);
		//inRange(InfraFiltered, 250, 255, InfraFiltered);
		InfraFiltered.convertTo(InfraFiltered, CV_8UC1);
		/*Mat erodeElement = getStructuringElement(MORPH_RECT, Size(2, 2));
		erode(InfraFiltered, InfraFiltered, erodeElement);*/
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(6, 6));
		dilate(InfraFiltered, InfraFiltered, dilateElement);
		vector< std::vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(InfraFiltered, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		Moments moment[5];
		CameraSpacePoint cameraPoint[5];
		DepthSpacePoint depthPoint[5];
		Point2i marker[5];
		if (hierarchy.size() == 5) {
			for (int i = 0; i < 5; i++) {
				marker[i] = Point2i(0, 0);
				moment[i] = moments((Mat)contours[i]);
				depthPoint[i].X = moment[i].m10 / moment[i].m00;
				depthPoint[i].Y = moment[i].m01 / moment[i].m00;
			}
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < 5; j++) {
					if (j != i) {
						for (int k = 0; k < 5; k++) {
							if (k != i && k != j) {
								for (int l = 0; l < 5; l++) {
									if (l != i && l != j && l != k) {
										for (int m = 0; m < 5; m++) {
											if (m != i && m != j && m != k && m != l) {
												DepthSpacePoint MeanIJ;
												CameraSpacePoint temp;
												MeanIJ.X = (depthPoint[i].X + depthPoint[j].X) / 2;
												MeanIJ.Y = (depthPoint[i].Y + depthPoint[j].Y) / 2;
												getMarker2Points(depthPoint[k], MeanIJ, &cameraPoint[k], &temp, 2, 0.1);
												getMarker2Points(depthPoint[i], depthPoint[k], &cameraPoint[i], &temp, 2, 0.1);
												getMarker2Points(depthPoint[j], depthPoint[k], &cameraPoint[j], &temp, 2, 0.1);
												getMarker2Points(depthPoint[l], depthPoint[m], &cameraPoint[l], &cameraPoint[m], 2, 0.1);
												float distanceA = sqrt(pow(cameraPoint[i].X - cameraPoint[j].X, 2) + pow(cameraPoint[i].Y - cameraPoint[j].Y, 2) + pow(cameraPoint[i].Z - cameraPoint[j].Z, 2));
												float distanceB = sqrt(pow(cameraPoint[k].X - cameraPoint[j].X, 2) + pow(cameraPoint[k].Y - cameraPoint[j].Y, 2) + pow(cameraPoint[k].Z - cameraPoint[j].Z, 2));
												float distanceC = sqrt(pow(cameraPoint[i].X - cameraPoint[k].X, 2) + pow(cameraPoint[i].Y - cameraPoint[k].Y, 2) + pow(cameraPoint[i].Z - cameraPoint[k].Z, 2));
												float distanceD = sqrt(pow(cameraPoint[l].X - cameraPoint[m].X, 2) + pow(cameraPoint[l].Y - cameraPoint[m].Y, 2) + pow(cameraPoint[l].Z - cameraPoint[m].Z, 2));												
												if (distanceA < 0.17 && distanceA > 0.1 && //14 A(ij) violin
													distanceB < 0.27 && distanceB > 0.2 && //23.5 B(kj) violin
													distanceC < 0.27 && distanceC > 0.2 && //23.5 C(ki) violin
													distanceD < 0.5 && distanceD > 0.35) { //45.5 D(lm) bow
													//printf("%0.2f:%0.2f:%0.2f:%0.2f\n", distanceA, distanceB, distanceC, distanceD);
													line(InfraFiltered, Point2i(depthPoint[k].X, depthPoint[k].Y), Point2i(depthPoint[j].X, depthPoint[j].Y), 255);
													line(InfraFiltered, Point2i(depthPoint[k].X, depthPoint[k].Y), Point2i(depthPoint[i].X, depthPoint[i].Y), 255);
													line(InfraFiltered, Point2i(depthPoint[l].X, depthPoint[l].Y), Point2i(depthPoint[m].X, depthPoint[m].Y), 255);
													for (int o = 0; o < 5; o++) {
														char text[20];
														sprintf(text, " %.2f : %.2f : %.2f", cameraPoint[o].X, cameraPoint[o].Y, cameraPoint[o].Z);
														String string = text;
														cv::putText(InfraFiltered, string, Point2i(depthPoint[o].X, depthPoint[o].Y), FONT_HERSHEY_PLAIN, 1.0, 80, 2.0);
													}
													float tilt = computeTilt(cameraPoint[k], cameraPoint[i], cameraPoint[j], cameraPoint[l], cameraPoint[m]);
													char text[20];
													smoothed_tilt = 0.9*smoothed_tilt + 0.1*tilt;
													sprintf(text, "Tilt = %d", (int)(smoothed_tilt / 3.1415926 *180));
													String string = text;
													cv::putText(InfraFiltered, string, Point2i(30,30), FONT_HERSHEY_PLAIN, 2.0, 80, 2.0);
													i = 5; j = 5; k = 5; l = 5; m = 5;
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		//cout << count << endl;
		show();
		const int key = waitKey(10);
		if (key == VK_ESCAPE) {
			break;
		}
	}
}

float Kinect::computeTilt(CameraSpacePoint scroll, CameraSpacePoint boutA, CameraSpacePoint boutB, CameraSpacePoint bowA, CameraSpacePoint bowB) {
	CameraSpacePoint violinA, violinB,violinVector,bowVector;
	violinA.X = scroll.X - boutA.X;
	violinA.Y = scroll.Y - boutA.Y;
	violinA.Z = scroll.Z - boutA.Z;
	violinB.X = scroll.X - boutB.X;
	violinB.Y = scroll.Y - boutB.Y;
	violinB.Z = scroll.Z - boutB.Z;
	violinVector.X = violinA.Y * violinB.Z - violinA.Z * violinB.Y;
	violinVector.Y = -(violinA.X*violinB.Z - violinA.Z*violinB.X);
	violinVector.Z = violinA.X*violinB.Y - violinA.Y*violinB.X;
	if (bowA.Y < bowB.Y) { //if the A is the upper point of the bow
		bowVector.X = bowA.X - bowB.X;
		bowVector.Y = bowA.Y - bowB.Y;
		bowVector.Z = bowA.Z - bowB.Z;
	}
	else {
		bowVector.X = bowB.X - bowA.X;
		bowVector.Y = bowB.Y - bowA.Y;
		bowVector.Z = bowB.Z - bowA.Z;
	}
	return asin((violinVector.X*bowVector.X + violinVector.Y*bowVector.Y + violinVector.Z*bowVector.Z) / 
		(sqrt(pow(violinVector.X, 2) + pow(violinVector.Y, 2) + pow(violinVector.Z, 2)) 
			* sqrt(pow(bowVector.X, 2) + pow(bowVector.Y, 2) + pow(bowVector.Z, 2))));
}

uint16_t Kinect::getMarkerDistance(DepthSpacePoint depthPoint, uint16_t radius) {
	uint16_t dist = 8000;
	for (int j = -radius; j <= radius; j++) {
		for (int k = -radius; k <= radius; k++) {
			uint16_t temp = depthMat.at<uint16_t>(Point2i(depthPoint.X - j, depthPoint.Y + k));
			if (temp > 499 && temp < dist) {
				dist = temp;
			}
		}
	}
	return dist;
}

void Kinect::getMarker2Points(DepthSpacePoint pointA, DepthSpacePoint pointB, CameraSpacePoint *camA, CameraSpacePoint *camB, uint8_t radius, float offset) {
	DepthSpacePoint A, B;
	uint16_t depthA, depthB;
	A.X = pointA.X + (pointB.X - pointA.X)*offset;
	A.Y = pointA.Y + (pointB.Y - pointA.Y)*offset;
	B.X = pointB.X + (pointA.X - pointB.X)*offset;
	B.Y = pointB.Y + (pointA.Y - pointB.Y)*offset;
	depthA = getMarkerDistance(A, 2);
	depthB = getMarkerDistance(B, 2);
	coordinateMapper->MapDepthPointToCameraSpace(A, depthA, camA);
}

// Initialize
void Kinect::initialize()
{
	setUseOptimized(true);

	// Initialize Sensor
	initializeSensor();

	// Initialize Multi Source
	initializeMultiSource();

	// Initialize Color
	initializeColor();

	// Initialize Depth
	initializeDepth();

	// Initialize Infrared
	initializeInfrared();


	// Wait a Few Seconds until begins to Retrieve Data from Sensor ( about 2000-[ms] )
	std::this_thread::sleep_for(std::chrono::seconds(2));
}

// Initialize Sensor
inline void Kinect::initializeSensor()
{
	// Open Sensor
	ERROR_CHECK(GetDefaultKinectSensor(&kinect));

	ERROR_CHECK(kinect->Open());

	// Check Open
	BOOLEAN isOpen = FALSE;
	ERROR_CHECK(kinect->get_IsOpen(&isOpen));
	if (!isOpen) {
		throw std::runtime_error("failed IKinectSensor::get_IsOpen( &isOpen )");
	}
	// Retrieve Coordinate Mapper
	ERROR_CHECK(kinect->get_CoordinateMapper(&coordinateMapper));
}

// Initialize Multi Source
inline void Kinect::initializeMultiSource()
{
	// Open Multi Source Reader
	DWORD types = FrameSourceTypes::FrameSourceTypes_Color
		| FrameSourceTypes::FrameSourceTypes_Depth;

	ERROR_CHECK(kinect->OpenMultiSourceFrameReader(types, &multiSourceFrameReader));
}

// Initialize Color
inline void Kinect::initializeColor()
{
	// Open Color Reader
	ComPtr<IColorFrameSource> colorFrameSource;
	ERROR_CHECK(kinect->get_ColorFrameSource(&colorFrameSource));

	// Retrieve Color Description
	ComPtr<IFrameDescription> colorFrameDescription;
	ERROR_CHECK(colorFrameSource->CreateFrameDescription(ColorImageFormat::ColorImageFormat_Bgra, &colorFrameDescription));
	ERROR_CHECK(colorFrameDescription->get_Width(&colorWidth)); // 1920
	ERROR_CHECK(colorFrameDescription->get_Height(&colorHeight)); // 1080
	ERROR_CHECK(colorFrameDescription->get_BytesPerPixel(&colorBytesPerPixel)); // 4

																				// Allocation Color Buffer
	colorBuffer.resize(colorWidth * colorHeight * colorBytesPerPixel);
}

// Initialize Depth
inline void Kinect::initializeDepth()
{
	// Open Depth Reader
	ComPtr<IDepthFrameSource> depthFrameSource;
	ERROR_CHECK(kinect->get_DepthFrameSource(&depthFrameSource));

	// Retrieve Depth Description
	ComPtr<IFrameDescription> depthFrameDescription;
	ERROR_CHECK(depthFrameSource->get_FrameDescription(&depthFrameDescription));
	ERROR_CHECK(depthFrameDescription->get_Width(&depthWidth)); // 512
	ERROR_CHECK(depthFrameDescription->get_Height(&depthHeight)); // 424
	ERROR_CHECK(depthFrameDescription->get_BytesPerPixel(&depthBytesPerPixel)); // 2

																				// Allocation Depth Buffer
	depthBuffer.resize(depthWidth * depthHeight);
}

inline void Kinect::initializeInfrared()
{
	// Open Infrared Reader
	ComPtr<IInfraredFrameSource> infraredFrameSource;
	ERROR_CHECK(kinect->get_InfraredFrameSource(&infraredFrameSource));
	ERROR_CHECK(infraredFrameSource->OpenReader(&infraredFrameReader));

	// Retrieve Infrared Description
	ComPtr<IFrameDescription> infraredFrameDescription;
	ERROR_CHECK(infraredFrameSource->get_FrameDescription(&infraredFrameDescription));
	ERROR_CHECK(infraredFrameDescription->get_Width(&infraredWidth)); // 512
	ERROR_CHECK(infraredFrameDescription->get_Height(&infraredHeight)); // 424
	ERROR_CHECK(infraredFrameDescription->get_BytesPerPixel(&infraredBytesPerPixel)); // 2

																					  // Allocation Depth Buffer
	infraredBuffer.resize(infraredWidth * infraredHeight);
}

// Finalize
void Kinect::finalize()
{
	destroyAllWindows();

	// Close Sensor
	if (kinect != nullptr) {
		kinect->Close();
	}
}

// Update Data
void Kinect::update()
{
	// Retrieve Multi Source Frame
	ComPtr<IMultiSourceFrame> multiSourceFrame;
	const HRESULT ret = multiSourceFrameReader->AcquireLatestFrame(&multiSourceFrame);
	if (FAILED(ret)) {
		return;
	}

	// Update Color
	updateColor(multiSourceFrame);

	// Update Depth
	updateDepth(multiSourceFrame);

	// Update Infrared
	updateInfrared();
}

// Update Color
inline void Kinect::updateColor(const ComPtr<IMultiSourceFrame>& multiSourceFrame)
{
	if (multiSourceFrame == nullptr) {
		return;
	}

	// Retrieve Color Frame Reference
	ComPtr<IColorFrameReference> colorFrameReference;
	HRESULT ret = multiSourceFrame->get_ColorFrameReference(&colorFrameReference);
	if (FAILED(ret)) {
		return;
	}

	// Retrieve Color Frame
	ComPtr<IColorFrame> colorFrame;
	ret = colorFrameReference->AcquireFrame(&colorFrame);
	if (FAILED(ret)) {
		return;
	}

	// Convert Format ( YUY2 -> BGRA )
	ERROR_CHECK(colorFrame->CopyConvertedFrameDataToArray(static_cast<UINT>(colorBuffer.size()), &colorBuffer[0], ColorImageFormat::ColorImageFormat_Bgra));
}

// Update Depth
inline void Kinect::updateDepth(const ComPtr<IMultiSourceFrame>& multiSourceFrame)
{
	if (multiSourceFrame == nullptr) {
		return;
	}

	// Retrieve Depth Frame Reference
	ComPtr<IDepthFrameReference> depthFrameReference;
	HRESULT ret = multiSourceFrame->get_DepthFrameReference(&depthFrameReference);
	if (FAILED(ret)) {
		return;
	}

	// Retrieve Depth Frame
	ComPtr<IDepthFrame> depthFrame;
	ret = depthFrameReference->AcquireFrame(&depthFrame);
	if (FAILED(ret)) {
		return;
	}

	// Retrieve Depth Data
	ERROR_CHECK(depthFrame->CopyFrameDataToArray(static_cast<UINT>(depthBuffer.size()), &depthBuffer[0]));
}

// Update Infrared
inline void Kinect::updateInfrared()
{
	// Retrieve Infrared Frame
	ComPtr<IInfraredFrame> infraredFrame;
	const HRESULT ret = infraredFrameReader->AcquireLatestFrame(&infraredFrame);
	if (FAILED(ret)) {
		return;
	}

	// Retrieve Infrared Data
	ERROR_CHECK(infraredFrame->CopyFrameDataToArray(static_cast<UINT>(infraredBuffer.size()), &infraredBuffer[0]));
}

// Draw Data
void Kinect::draw()
{
	// Draw Color
	drawColor();

	// Draw Depth
	drawDepth();

	// Draw Infrared
	drawInfrared();
}

// Draw Color
inline void Kinect::drawColor()
{
	// Create Mat from Color Buffer
	colorMat = Mat(colorHeight, colorWidth, CV_8UC4, &colorBuffer[0]);
}

// Draw Depth
inline void Kinect::drawDepth()
{
	// Create Mat from Depth Buffer
	depthMat = Mat(depthHeight, depthWidth, CV_16UC1, &depthBuffer[0]);
}

// Draw Infrared
inline void Kinect::drawInfrared()
{
	// Create Mat from Infrared Buffer
	infraredMat = Mat(infraredHeight, infraredWidth, CV_16UC1, &infraredBuffer[0]);
}

// Show Data
void Kinect::show()
{
	// Show Color
	showColor();

	// Show Depth
	showDepth();

	// Show Infrared
	showInfrared();
}

// Show Color
inline void Kinect::showColor()
{
	if (colorMat.empty()) {
		return;
	}

	// Resize Image
	Mat resizeMat;
	const double scale = 0.5;
	resize(colorMat, resizeMat, Size(), scale, scale);

	// Show Image
	imshow("Color", resizeMat);
}

// Show Depth
inline void Kinect::showDepth()
{
	if (depthMat.empty()) {
		return;
	}

	// Scaling ( 0-8000 -> 255-0 )
	Mat scaleMat;
	depthMat.convertTo(scaleMat, CV_8U, -255.0 / 8000.0, 255.0);
	//applyColorMap( scaleMat, scaleMat, COLORMAP_BONE );

	// Show Image
	imshow("Depth", scaleMat);
}

inline void Kinect::showInfrared()
{
	if (infraredMat.empty()) {
		return;
	}

	// Scaling ( 0b1111'1111'0000'0000 -> 0b1111'1111 )
	/*Mat scaleMat(infraredHeight, infraredWidth, CV_8UC1);
	scaleMat.forEach<uchar>([&](uchar &p, const int* position) {
		p = infraredMat.at<ushort>(position[0], position[1]) >> 8;
	});*/

	// Show Image
	imshow("InfraredFiltered", InfraFiltered);
	imshow("Infrared", infraredMat);
}