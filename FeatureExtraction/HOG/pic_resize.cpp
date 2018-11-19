#include"SelfHog.h"
using namespace std;
using namespace cv;

int PIC_RESIZE() 
{
	string load_path = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\original_true\\",
		save_path = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\true\\";
	int resize_height = IMG_HEIGHT;
	int resize_width = IMG_WIDTH;
	for (int i = 1; i <= 20; ++i)
	{
		Mat src = cv::imread(load_path+to_string(i)+".jpg"), dst;
		resize(src, dst, cv::Size(resize_width, resize_height), (0, 0), (0, 0), cv::INTER_LINEAR);
		imwrite(save_path+to_string(i)+".jpg", dst);
	}
	return 0;
}