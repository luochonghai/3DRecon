#include"stdafx.h"
#include "calib.h"
#include "undistort.h"

using namespace std;
using namespace cv;

int main()
{
	string patternImgPath = "data/pattern/";
	string calibResultPath = "D:\\FDU\\Tracking\\";
	string srcImgPath = "data/srcImg/0.jpg";
	Size boardSize = Size(NUM_WIDTH, NUM_HEIGHT);
	CCalibration calibration(patternImgPath, calibResultPath, boardSize);
	calibration.run();
	CUndistort undistort(srcImgPath, calibResultPath);
	undistort.run();
}