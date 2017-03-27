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
	float smoothed_tilt;
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
	float computeTilt(CameraSpacePoint scroll, CameraSpacePoint boutA, CameraSpacePoint boutB, CameraSpacePoint bowA, CameraSpacePoint bowB);
	void getMarker2Points(DepthSpacePoint pointA, DepthSpacePoint pointB, CameraSpacePoint *distA, CameraSpacePoint *distB, uint8_t radius, float offset);
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