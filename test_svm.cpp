#include"SelfHog.h"
using namespace std;
using namespace cv;
using namespace cv::ml;
string test_path = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\test\\11.jpg";

void SVM_HOG_detect()
{
	Ptr<SVM> svm = StatModel::load<SVM>("SVM_HOG.xml"); 

	if (svm->empty()) { //empty()���� �ַ����ǿյĻ�������true
		cout << "��ȡXML�ļ�ʧ�ܡ�" << endl;
		return;
	}
	else {
		cout << "��ȡXML�ļ��ɹ���" << endl;
	}


	Mat svecsmat = svm->getSupportVectors();//svecsmatԪ�ص���������Ϊfloat

	int svdim = svm->getVarCount();

	int numofsv = svecsmat.rows;

	Mat alphamat = Mat::zeros(numofsv, svdim, CV_32F);//alphamat��svindex�����ʼ��������getDecisionFunction()�����ᱨ��
	Mat svindex = Mat::zeros(1, numofsv, CV_64F);

	Mat Result;
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	alphamat.convertTo(alphamat, CV_32F);//��alphamatԪ�ص�������������ת��CV_32F

	cout << "1" << endl;
	Result = -1 * alphamat * svecsmat;//float
	cout << "2" << endl;

	vector<float> vec;
	for (int i = 0; i < svdim; ++i)
	{
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back(rho);

	//����HOG�����ļ�
	ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i < vec.size(); ++i)
	{
		fout << vec[i] << endl;
	}
	cout << "�������" << endl;

	//----------��ȡͼƬ���м��----------------------------
	//  HOGDescriptor hog_test;
	HOGDescriptor hog_test(Size(IMG_WIDTH, IMG_HEIGHT), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	hog_test.setSVMDetector(vec);

	Mat src = imread(test_path, 0);
	if (!src.data) {
		cout << "����ͼƬ��ȡʧ��" << endl;
		return;
	}
	vector<Rect> found, found_filtered;

	int p = 1;
	resize(src, src, Size(src.cols / p, src.rows / p));

	clock_t startTime, finishTime;
	cout << "��ʼ���" << endl;

	startTime = clock();                                                //1.05
	hog_test.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);   //��߶ȼ��
	finishTime = clock();
	cout << "�������ʱ��Ϊ" << (finishTime - startTime)*1.0 / CLOCKS_PER_SEC << " �� " << endl;

	cout << endl << "���ο�ĳߴ�Ϊ : " << found.size() << endl;

	//�ҳ�����û��Ƕ�׵ľ���,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����η���found_filtered��
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
	cout << endl << "Ƕ�׾��ο�ϲ����" << endl;

	//�����ο���Ϊhog�����ľ��ο��ʵ�ʵĿ�Ҫ��΢��Щ,����������Ҫ��һЩ����
	for (int i = 0; i<found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];

		r.x += cvRound(r.width*0.1); //int cvRound(double value) ��һ��double�͵��������������룬������һ����������
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