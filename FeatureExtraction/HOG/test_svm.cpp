#include"SelfHog.h"
using namespace std;
using namespace cv;
using namespace cv::ml;
string test_path = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\test\\11.jpg";

void SVM_HOG_detect()
{
	Ptr<SVM> svm = StatModel::load<SVM>("SVM_HOG.xml"); 

	if (svm->empty()) { //empty()函数 字符串是空的话返回是true
		cout << "读取XML文件失败。" << endl;
		return;
	}
	else {
		cout << "读取XML文件成功。" << endl;
	}


	Mat svecsmat = svm->getSupportVectors();//svecsmat元素的数据类型为float

	int svdim = svm->getVarCount();

	int numofsv = svecsmat.rows;

	Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错
	Mat svindex = Mat::zeros(1, numofsv, CV_64F);

	Mat Result;
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	alphamat.convertTo(alphamat, CV_32F);//将alphamat元素的数据类型重新转成CV_32F

	cout << "1" << endl;
	Result = -1 * alphamat * svecsmat;//float
	cout << "2" << endl;

	vector<float> vec;
	for (int i = 0; i < svdim; ++i)
	{
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back(rho);

	//保存HOG检测的文件
	ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i < vec.size(); ++i)
	{
		fout << vec[i] << endl;
	}
	cout << "保存完毕" << endl;

	//----------读取图片进行检测----------------------------
	//  HOGDescriptor hog_test;
	HOGDescriptor hog_test(Size(IMG_WIDTH, IMG_HEIGHT), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog_test.setSVMDetector(vec);

	Mat src = imread(test_path, 0);
	if (!src.data) {
		cout << "测试图片读取失败" << endl;
		return;
	}
	vector<Rect> found, found_filtered;

	int p = 1;
	resize(src, src, Size(src.cols / p, src.rows / p));

	clock_t startTime, finishTime;
	cout << "开始检测" << endl;

	startTime = clock();                                                //1.05
	hog_test.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);   //多尺度检测
	finishTime = clock();
	cout << "检测所用时间为" << (finishTime - startTime)*1.0 / CLOCKS_PER_SEC << " 秒 " << endl;

	cout << endl << "矩形框的尺寸为 : " << found.size() << endl;

	//找出所有没有嵌套的矩形,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形放入found_filtered中
	for (int i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j = 0;
		for (; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}
	cout << endl << "嵌套矩形框合并完毕" << endl;

	//画矩形框，因为hog检测出的矩形框比实际的框要稍微大些,所以这里需要做一些调整
	for (int i = 0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];

		r.x += cvRound(r.width*0.1); //int cvRound(double value) 对一个double型的数进行四舍五入，并返回一个整型数！
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);

		rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
	}

	imwrite("D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\true\\ImgProcessed.jpg", src);
	namedWindow("src", 0);
	imshow("src", src);
	waitKey(0);
}