#include"SelfHog.h"

using namespace std;
using namespace cv;

int imageCount = 0; //�ü������ĸ�����ͼƬ����  
int FalseSampleGen()
{
	Mat src;
	string ImgName;
	string readAddress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\original_false\\";
	string saveAddress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\false\\";
	string saveName;//�ü������ĸ�����ͼƬ�ļ���  
	ifstream fin(readAddress + "NegativeSample.txt");//��ԭʼ������ͼƬ�ļ��б�  

													 //һ��һ�ж�ȡ�ļ��б�  
	while (getline(fin, ImgName))
	{
		ImgName = readAddress + ImgName;

		src = imread(ImgName);//��ȡͼƬ  
		int originalWidth = src.cols;
		int originalHeight = src.rows;
		int width = originalWidth / 4;
		int height = originalHeight / 4;
		resize(src, src, Size(width, height)); //��ͼƬ�ߴ�ѹ�����Ի�ȡ������Ϣ

											   //ͼƬ��СӦ���������ٰ���һ��64*128�Ĵ���  
		if (src.cols >= IMG_WIDTH && src.rows >= IMG_HEIGHT) //ͼƬ�ߴ��С����Ҫ��
		{
			srand(time(NULL));//�������������  

							  //��ÿ��ͼƬ������ü�200��64*128��С�ĸ�����  
			for (int i = 0; i < SAMPLE_NUM; i++)
			{
				int x = (rand() % (src.cols - IMG_WIDTH)); //���Ͻ�x����   rand()%a �ܹ��õ�0��a�ڵ������
				int y = (rand() % (src.rows - IMG_HEIGHT)); //���Ͻ�y����  
				Mat imgROI = src(Rect(x, y, IMG_WIDTH, IMG_HEIGHT));

				saveName = saveAddress + to_string(++imageCount) + ".jpg";
				imwrite(saveName, imgROI);//�����ļ�  

				if (imageCount % 10 == 0) //ÿ����10��ͼƬ���һ������
				{
					system("cls");
					cout << endl << "            ԭʼͼ��ߴ磺 " << originalWidth << " * " << originalHeight << endl;
					cout << "           resize��ͼ��ߴ磺 " << width << " * " << height << endl;
					cout << endl << "           �Ѳü�ͼƬ������ " << imageCount << endl;
				}
			}
		}

		//break; //--------------
	}

	//system("pause");

	return 0;
}