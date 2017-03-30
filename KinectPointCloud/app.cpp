#include "app.h"
#include "util.h"
#include <vector>
#include <thread>
#include <chrono>
#include <inttypes.h>
#include <math.h>
#include "opencv2/highgui/highgui.hpp"
#define SMOOTH 0.8
#define SPEED_SMOOTH 0.7
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
	smoothed_dist = 0;
	smoothed_paralel = 0;
	bowVioDist = 0;
	time = 0;
	bowPos = 0;
	previousBowPos = 0;
}

// Destructor
Kinect::~Kinect()
{
	// Finalize
	finalize();
}

// Processing
void Kinect::run()
{
	// Main Loop
	while (true) {
		//cv::putText(colorMat, to_string(getTickCount()), Point2i(30, 200), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
		
		update();
		draw();
		infraredMat.convertTo(InfraFiltered, CV_32FC1);
		threshold(InfraFiltered, InfraFiltered, 32000, 255, CV_THRESH_BINARY);
		InfraFiltered.convertTo(InfraFiltered, CV_8UC1);
		/*Mat erodeElement = getStructuringElement(MORPH_RECT, Size(2, 2));
		erode(InfraFiltered, InfraFiltered, erodeElement);*/
		Mat dilateElement = getStructuringElement(MORPH_RECT, Size(6, 6));
		dilate(InfraFiltered, InfraFiltered, dilateElement);
		vector< std::vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(InfraFiltered, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		Moments moment[5];
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
												float offset = 0.07;
												int searchRadius = 3;
												getMarker2Points(depthPoint[k], MeanIJ, &cameraPoint[k], &temp, searchRadius, offset);
												getMarker2Points(depthPoint[i], depthPoint[k], &cameraPoint[i], &temp, searchRadius, offset);
												getMarker2Points(depthPoint[j], depthPoint[k], &cameraPoint[j], &temp, searchRadius, offset);
												getMarker2Points(depthPoint[l], depthPoint[m], &cameraPoint[l], &cameraPoint[m], searchRadius, offset);
												float distanceA = sqrt(pow(cameraPoint[i].X - cameraPoint[j].X, 2) + pow(cameraPoint[i].Y - cameraPoint[j].Y, 2) + pow(cameraPoint[i].Z - cameraPoint[j].Z, 2));
												float distanceB = sqrt(pow(cameraPoint[k].X - cameraPoint[j].X, 2) + pow(cameraPoint[k].Y - cameraPoint[j].Y, 2) + pow(cameraPoint[k].Z - cameraPoint[j].Z, 2));
												float distanceC = sqrt(pow(cameraPoint[i].X - cameraPoint[k].X, 2) + pow(cameraPoint[i].Y - cameraPoint[k].Y, 2) + pow(cameraPoint[i].Z - cameraPoint[k].Z, 2));
												float distanceD = sqrt(pow(cameraPoint[l].X - cameraPoint[m].X, 2) + pow(cameraPoint[l].Y - cameraPoint[m].Y, 2) + pow(cameraPoint[l].Z - cameraPoint[m].Z, 2));												
												if (distanceA < 0.17 && distanceA > 0.11 && //14 A(ij) violin
													distanceB < 0.28 && distanceB > 0.22 && //25 B(kj) violin
													distanceC < 0.28 && distanceC > 0.22 && //25 C(ki) violin
													distanceD < 0.52 && distanceD > 0.40 && //49 D(lm) bow
													cameraPoint[i].Y >= cameraPoint[j].Y && //i will be up in the violin
													cameraPoint[l].Y >= cameraPoint[m].Y  //l will be up in the bow
													) { //45.5 D(lm) bow
													//printf("%0.2f:%0.2f:%0.2f:%0.2f\n", distanceA, distanceB, distanceC, distanceD);
													
													bowSpeed(m);

													ColorSpacePoint colorPoint[5];
													coordinateMapper->MapCameraPointsToColorSpace(5, cameraPoint, 5, colorPoint);
													
													line(colorMat, Point2i(colorPoint[i].X, colorPoint[i].Y), Point2i(colorPoint[j].X, colorPoint[j].Y), 255, 10);
													line(colorMat, Point2i(colorPoint[l].X, colorPoint[l].Y), Point2i(colorPoint[m].X, colorPoint[m].Y), 255, 10);
													/*for (int o = 0; o < 5; o++) {
														char text[20];
														sprintf(text, " %.2f : %.2f : %.2f", cameraPoint[o].X, cameraPoint[o].Y, cameraPoint[o].Z);
														String string = text;
														cv::putText(InfraFiltered, string, Point2i(depthPoint[o].X, depthPoint[o].Y), FONT_HERSHEY_PLAIN, 1.0, 80, 2.0);
													}*/
													CameraSpacePoint violinVector;
													smoothed_tilt = SMOOTH*smoothed_tilt + (1 - SMOOTH)* 57.29578 * computeTilt(cameraPoint[k], cameraPoint[i], cameraPoint[j], cameraPoint[l], cameraPoint[m], &violinVector);
													
													cv::putText(colorMat, "Angle in degrees: " + to_string(smoothed_tilt), Point2i(30, 130), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
													CameraSpacePoint bridge,tempC;
													ColorSpacePoint bridgeColor;
													tempC.X = (cameraPoint[i].X + cameraPoint[j].X) / 2;
													tempC.Y = (cameraPoint[i].Y + cameraPoint[j].Y) / 2;
													tempC.Z = (cameraPoint[i].Z + cameraPoint[j].Z) / 2;
													float bf = 0.62;
													bridge.X = tempC.X + bf*(tempC.X - cameraPoint[k].X);
													bridge.Y = tempC.Y + bf*(tempC.Y - cameraPoint[k].Y);
													bridge.Z = tempC.Z + bf*(tempC.Z - cameraPoint[k].Z);
													coordinateMapper->MapCameraPointToColorSpace(bridge, &bridgeColor);
													circle(colorMat, cvPoint(bridgeColor.X, bridgeColor.Y), 1/tempC.Z*5, 150, 5);
													line(colorMat, Point2i(colorPoint[k].X, colorPoint[k].Y), Point2i(bridgeColor.X, bridgeColor.Y), 255, 10);
													CameraSpacePoint bowLprojection, bowMprojection;
													ColorSpacePoint bowLpr, bowMpr;
													
													bowLprojection = projectionPoint2Plane(violinVector, cameraPoint[i], cameraPoint[l]);
													bowMprojection = projectionPoint2Plane(violinVector, cameraPoint[i], cameraPoint[m]); 
													
													coordinateMapper->MapCameraPointToColorSpace(bowLprojection, &bowLpr);
													coordinateMapper->MapCameraPointToColorSpace(bowMprojection, &bowMpr);

													//line(colorMat, Point2i(bowLpr.X, bowLpr.Y), Point2i(bowMpr.X, bowMpr.Y), 0, 10); 
													smoothed_paralel = /*smoothed_paralel*SMOOTH + (1- SMOOTH)**/57.29578 * angleBetweenLines3d(bridge, cameraPoint[k], cameraPoint[l], cameraPoint[m]);
													cv::putText(colorMat, "Bow angle with strings in degrees: " + to_string(smoothed_paralel), Point2i(30, 270), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
													
													smoothed_dist = smoothed_dist*SMOOTH +(1- SMOOTH)*1000 * distanceLine2point(bowLprojection, bowMprojection, bridge);
													cv::putText(colorMat, "Bow-Bridge distance in mm: " + to_string(smoothed_dist), Point2i(30, 200), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
													bowVioDist = SMOOTH * bowVioDist + (1-SMOOTH)*1000*distanceBetweenLines3d(bridge, cameraPoint[k], cameraPoint[l], cameraPoint[m]);
													cv::putText(colorMat, "Bow-violin distance in mm: " + to_string(bowVioDist), Point2i(30, 340), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
													cv::putText(colorMat, "Bow Speed in cm/sec: " + to_string((int)(100 * bow_speed)), Point2i(30, 410), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);

													if (bowVioDist <= 40) {// 8 -4 -13
														if(smoothed_tilt <= -13)
															cv::putText(colorMat, "Playing string E", Point2i(30, 480), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
														else if (smoothed_tilt <= -4)
															cv::putText(colorMat, "Playing string A", Point2i(30, 480), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
														else if (smoothed_tilt <= 8)
															cv::putText(colorMat, "Playing string D", Point2i(30, 480), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
														else 
															cv::putText(colorMat, "Playing string G", Point2i(30, 480), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);

													}
													else
														cv::putText(colorMat, "Bow on AIR", Point2i(30, 480), FONT_HERSHEY_PLAIN, 4.0, 80, 2.0);
													
													
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
void Kinect::savePreviousFrame() {

	for (int i = 0; i < 5; i++) {
		PrcameraPoint[i] = cameraPoint[i];
	}
	
	//cout << elapsedTime << endl;
}
void Kinect::bowSpeed(uint8_t bowP) {
	
	if ((getTickCount() - time) / getTickFrequency() > 0.04)
	{
		CameraSpacePoint dif;
		bowPos = bowP;
		dif = points2vector(cameraPoint[bowPos], PrcameraPoint[previousBowPos]);
		float dist = normV(dif);
		prTime = time;
		time = getTickCount();
		elapsedTime = (time - prTime) / getTickFrequency();
		bow_speed = bow_speed*SPEED_SMOOTH + (1-SPEED_SMOOTH)*dist / elapsedTime;
		savePreviousFrame();
		previousBowPos = bowPos;
	}	
}

CameraSpacePoint Kinect::points2vector(CameraSpacePoint a, CameraSpacePoint b) {
	CameraSpacePoint k;
	k.X = a.X - b.X;
	k.Y = a.Y - b.Y;
	k.Z = a.Z - b.Z;
	return k;
}
float Kinect::dotProduct(CameraSpacePoint a, CameraSpacePoint b) {
	return (a.X*b.X + a.Y*b.Y + a.Z*b.Z);
}
CameraSpacePoint Kinect::crossProduct(CameraSpacePoint a, CameraSpacePoint b) {
	CameraSpacePoint cross;
	cross.X = a.Y*b.Z - a.Z*b.Y;
	cross.Y = a.Z*b.X - a.X*b.Z;
	cross.Z = a.X*b.Y - a.Y*b.X;
	return cross;
}
CameraSpacePoint Kinect::normalizeVector(CameraSpacePoint a) {
	float norma = normV(a);
	a.X /= norma;
	a.Y /= norma;
	a.Z	/= norma;
	return a;
}
float Kinect::angleBetweenLines3d(CameraSpacePoint a1, CameraSpacePoint a2, CameraSpacePoint b1, CameraSpacePoint b2) {
	CameraSpacePoint lineA, lineB;
	lineA = normalizeVector(points2vector(a2, a1));
	lineB = normalizeVector(points2vector(b2, b1));
	return acos(dotProduct(lineA, lineB));
}
float Kinect::distanceBetweenLines3d(CameraSpacePoint a1, CameraSpacePoint a2, CameraSpacePoint b1, CameraSpacePoint b2) {
	CameraSpacePoint DA, DB;
	DA = points2vector(a2, a1);
	DB = points2vector(b2, b1);
	CameraSpacePoint cross = crossProduct(DA, DB);
	float crossNorm = normV(cross);
	CameraSpacePoint A;
	A.X = cross.X / crossNorm;
	A.Y = cross.Y / crossNorm;
	A.Z = cross.Z / crossNorm;
	CameraSpacePoint B = points2vector(a1, b1);
	float AB = dotProduct(A, B);
	return abs(AB);
}
float Kinect::normV(CameraSpacePoint v) {
	return sqrt(v.X*v.X + v.Y*v.Y + v.Z*v.Z);
}
CameraSpacePoint Kinect::projectionPoint2Plane(CameraSpacePoint plane,CameraSpacePoint planePoint, CameraSpacePoint point) {
	float t = (plane.X*planePoint.X - plane.X*point.X +
		plane.Y*planePoint.Y - plane.Y*point.Y +
		plane.Z*planePoint.Z - plane.Z*point.Z)
		/
		(plane.X*plane.X + plane.Y*plane.Y + plane.Z*plane.Z);
	CameraSpacePoint projection;
	projection.X = plane.X*t + point.X;
	projection.Y = plane.Y*t + point.Y;
	projection.Z = plane.Z*t + point.Z;
	return projection;
}
float Kinect::distanceLine2point(CameraSpacePoint lineA, CameraSpacePoint lineB, CameraSpacePoint point) {
	CameraSpacePoint x1 = lineA;
	CameraSpacePoint x2 = lineB;
	CameraSpacePoint x0 = point;
	CameraSpacePoint x10, x21, x1021;
	x10.X = x1.X - x0.X; x10.Y = x1.Y - x0.Y; x10.Z = x1.Z - x0.Z;
	x21.X = x2.X - x1.X; x21.Y = x2.Y - x1.Y; x21.Z = x2.Z - x1.Z;

	return
		sqrt(((x10.X*x10.X + x10.Y*x10.Y + x10.Z*x10.Z) * (x21.X*x21.X + x21.Y*x21.Y + x21.Z*x21.Z)
			- pow((x10.X*x21.X + x10.Y*x21.Y + x10.Z*x21.Z), 2))
		/
		(x21.X*x21.X + x21.Y*x21.Y + x21.Z*x21.Z));

}

float Kinect::computeTilt(CameraSpacePoint scroll, CameraSpacePoint boutA, CameraSpacePoint boutB, CameraSpacePoint bowA, CameraSpacePoint bowB,CameraSpacePoint *violinVector) {
	CameraSpacePoint violinA, violinB,bowVector;
	violinA.X = scroll.X - boutA.X;
	violinA.Y = scroll.Y - boutA.Y;
	violinA.Z = scroll.Z - boutA.Z;
	violinB.X = scroll.X - boutB.X;
	violinB.Y = scroll.Y - boutB.Y;
	violinB.Z = scroll.Z - boutB.Z;
	violinVector->X = violinA.Y * violinB.Z - violinA.Z * violinB.Y;
	violinVector->Y = -(violinA.X*violinB.Z - violinA.Z*violinB.X);
	violinVector->Z = violinA.X*violinB.Y - violinA.Y*violinB.X;
	/*float norm = sqrt(violinVector->X*violinVector->X + violinVector->Y*violinVector->Y + violinVector->Z*violinVector->Z);
	violinVector->X = violinVector->X / norm;
	violinVector->Y = violinVector->Y / norm;
	violinVector->Z = violinVector->Z / norm;*/
	//*d = -(violinVector->X * scroll.X + violinVector->Y * scroll.Y + violinVector->Y * scroll.Y);
	//float d2 = -(violinVector->X * boutA.X + violinVector->Y * boutA.Y + violinVector->Y * boutA.Y);
	////if (*d != -(violinVector->X * boutA.X + violinVector->Y * boutA.Y + violinVector->Y * boutA.Y))
	//	cout << *d <<endl;
	//	*d = d2;

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
	return asin((violinVector->X*bowVector.X + violinVector->Y*bowVector.Y + violinVector->Z*bowVector.Z) / 
		(sqrt(pow(violinVector->X, 2) + pow(violinVector->Y, 2) + pow(violinVector->Z, 2)) 
			* sqrt(pow(bowVector.X, 2) + pow(bowVector.Y, 2) + pow(bowVector.Z, 2))));
}



uint16_t Kinect::getMarkerDistance(DepthSpacePoint depthPoint, uint16_t radius) {
	uint16_t dist = 8000, avg = 0, count = 0,jitter=100;
	for (int j = -radius; j <= radius; j++) {
		for (int k = -radius; k <= radius; k++) {
			uint16_t temp = depthMat.at<uint16_t>(Point2i(depthPoint.X - j, depthPoint.Y + k));
			if (temp > 499 && temp < dist) {
				dist = temp;
			}
		}
	}
	return dist;
	//
	//for (int j = -radius; j <= radius; j++) {
	//	for (int k = -radius; k <= radius; k++) {
	//		uint16_t temp = depthMat.at<uint16_t>(Point2i(depthPoint.X - j, depthPoint.Y + k));
	//		if (temp <= dist + jitter && temp >=dist) {
	//			avg += temp;
	//			count++;
	//		}
	//	}
	//}
	////cout << count << "\t"<< avg / count << endl;
	//return avg / count;
}

void Kinect::getMarker2Points(DepthSpacePoint pointA, DepthSpacePoint pointB, CameraSpacePoint *camA, CameraSpacePoint *camB, uint8_t radius, float offset) {
	DepthSpacePoint A, B;
	uint16_t depthA, depthB;
	A.X = pointA.X + (pointB.X - pointA.X)*offset;
	A.Y = pointA.Y + (pointB.Y - pointA.Y)*offset;
	B.X = pointB.X + (pointA.X - pointB.X)*offset;
	B.Y = pointB.Y + (pointA.Y - pointB.Y)*offset;
	depthA = getMarkerDistance(A, radius);
	depthB = getMarkerDistance(B, radius);
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
	//showDepth();

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
	const double scale = 0.7;
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