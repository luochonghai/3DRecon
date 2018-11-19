#include"SelfHog.h"

using namespace std;
using namespace cv;

int imageCount = 0; //裁剪出来的负样本图片个数  
int FalseSampleGen()
{
	Mat src;
	string ImgName;
	string readAddress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\original_false\\";
	string saveAddress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\false\\";
	string saveName;//裁剪出来的负样本图片文件名  
	ifstream fin(readAddress + "NegativeSample.txt");//打开原始负样本图片文件列表  

													 //一行一行读取文件列表  
	while (getline(fin, ImgName))
	{
		ImgName = readAddress + ImgName;

		src = imread(ImgName);//读取图片  
		int originalWidth = src.cols;
		int originalHeight = src.rows;
		int width = originalWidth / 4;
		int height = originalHeight / 4;
		resize(src, src, Size(width, height)); //将图片尺寸压缩，以获取更多信息

											   //图片大小应该能能至少包含一个64*128的窗口  
		if (src.cols >= IMG_WIDTH && src.rows >= IMG_HEIGHT) //图片尺寸大小满足要求
		{
			srand(time(NULL));//设置随机数种子  

							  //从每张图片中随机裁剪200个64*128大小的负样本  
			for (int i = 0; i < SAMPLE_NUM; i++)
			{
				int x = (rand() % (src.cols - IMG_WIDTH)); //左上角x坐标   rand()%a 能够得到0到a内的随机数
				int y = (rand() % (src.rows - IMG_HEIGHT)); //左上角y坐标  
				Mat imgROI = src(Rect(x, y, IMG_WIDTH, IMG_HEIGHT));

				saveName = saveAddress + to_string(++imageCount) + ".jpg";
				imwrite(saveName, imgROI);//保存文件  

				if (imageCount % 10 == 0) //每生成10张图片输出一次数据
				{
					system("cls");
					cout << endl << "            原始图像尺寸： " << originalWidth << " * " << originalHeight << endl;
					cout << "           resize后图像尺寸： " << width << " * " << height << endl;
					cout << endl << "           已裁剪图片数量： " << imageCount << endl;
				}
			}
		}

		//break; //--------------
	}

	//system("pause");

	return 0;
}