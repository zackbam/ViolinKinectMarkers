#ifndef __APP__
#define __APP__

#include <Windows.h>
#include <Kinect.h>
#include <opencv2/opencv.hpp>

#include <vector>

#include <wrl/client.h>
using namespace Microsoft::WRL;

class Kinect
{
private:
	// Sensor
	ComPtr<IKinectSensor> kinect;

	// Reader
	ComPtr<IMultiSourceFrameReader> multiSourceFrameReader;// Reader
	ComPtr<IInfraredFrameReader> infraredFrameReader;
	ComPtr<ICoordinateMapper> coordinateMapper;
	// Color Buffer
	std::vector<BYTE> colorBuffer;
	int colorWidth;
	int colorHeight;
	int smoothed_tilt;
	int smoothed_dist;
	int smoothed_paralel;
	int bowVioDist;
	int64 time, prTime;
	double elapsedTime,bow_speed;
	uint8_t previousBowPos,bowPos;
	CameraSpacePoint cameraPoint[5];
	CameraSpacePoint PrcameraPoint[5];
	unsigned int colorBytesPerPixel;
	cv::Mat colorMat;
	// Depth Buffer
	std::vector<UINT16> depthBuffer;
	int depthWidth;
	int depthHeight;
	unsigned int depthBytesPerPixel;
	cv::Mat depthMat;
	
	// Infrared Buffer
	std::vector<UINT16> infraredBuffer;
	int infraredWidth;
	int infraredHeight;
	unsigned int infraredBytesPerPixel;
	cv::Mat infraredMat;
	cv::Mat InfraFiltered;
public:
	// Constructor
	Kinect();

	// Destructor
	~Kinect();

	// Processing
	void run();

private:
	// Initialize
	void initialize();
	uint16_t getMarkerDistance(DepthSpacePoint depthPoint, uint16_t radius);
	float computeTilt(CameraSpacePoint scroll, CameraSpacePoint boutA, CameraSpacePoint boutB, CameraSpacePoint bowA, CameraSpacePoint bowB, CameraSpacePoint * violinVector);
	float distanceLine2point(CameraSpacePoint lineA, CameraSpacePoint lineB, CameraSpacePoint point);
	CameraSpacePoint projectionPoint2Plane(CameraSpacePoint plane,CameraSpacePoint planePoint, CameraSpacePoint point);
	void getMarker2Points(DepthSpacePoint pointA, DepthSpacePoint pointB, CameraSpacePoint *distA, CameraSpacePoint *distB, uint8_t radius, float offset);
	float normV(CameraSpacePoint v);
	float angleBetweenLines3d(CameraSpacePoint a1, CameraSpacePoint a2, CameraSpacePoint b1, CameraSpacePoint b2);
	float distanceBetweenLines3d(CameraSpacePoint a1, CameraSpacePoint a2, CameraSpacePoint b1, CameraSpacePoint b2);
	CameraSpacePoint points2vector(CameraSpacePoint a, CameraSpacePoint b);
	float dotProduct(CameraSpacePoint a, CameraSpacePoint b);
	CameraSpacePoint normalizeVector(CameraSpacePoint a);
	CameraSpacePoint crossProduct(CameraSpacePoint a, CameraSpacePoint b);
	void bowSpeed(uint8_t bowPos);
	void savePreviousFrame();
	// Initialize Sensor
	inline void initializeSensor();
	
	// Initialize Multi Source
	inline void initializeMultiSource();

	// Initialize Color
	inline void initializeColor();

	// Initialize Depth
	inline void initializeDepth();

	// Initialize Infrared
	inline void initializeInfrared();

	// Finalize
	void finalize();

	// Update Data
	void update();

	// Update Color
	inline void updateColor(const ComPtr<IMultiSourceFrame>& multiSourceFrame);

	// Update Depth
	inline void updateDepth(const ComPtr<IMultiSourceFrame>& multiSourceFrame);

	// Update Infrared
	inline void updateInfrared();

	// Draw Data
	void draw();

	// Draw Color
	inline void drawColor();

	// Draw Depth
	inline void drawDepth();

	// Draw Infrared
	inline void drawInfrared();

	// Show Data
	void show();

	// Show Color
	inline void showColor();

	// Show Depth
	inline void showDepth();

	// Show Infrared
	inline void showInfrared();
};

#endif // __APP__