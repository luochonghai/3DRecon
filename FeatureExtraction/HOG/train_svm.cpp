#include"SelfHog.h"
using namespace std;
using namespace cv;
using namespace cv::ml;

void train_SVM_HOG()
{

	HOGDescriptor hog(Size(IMG_WIDTH, IMG_HEIGHT), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	int DescriptorDim; //HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	//  svm->setC(0.01); //���óͷ�����C��Ĭ��ֵΪ1
	svm->setKernel(SVM::LINEAR); //���Ժ�
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6)); //3000�ǵ���������1e-6�Ǿ�ȷ�ȡ�
																			//setTermCriteria�����������㷨����ֹ������ 
	//SVMѵ���Ĺ��̾���һ��ͨ�� ���� ��ʽ���Լ�������µĶ����Ż����⣬��������ָ��һ��������������������
	//��������㷨���ʵ���������ֹͣ����


	string ImgName;//ͼƬ������
	string PosSampleAdress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\true\\";
	string NegSampleAdress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\false\\";
	string HardSampleAdress = "";

	ifstream finPos(PosSampleAdress + "PosSample.txt"); //��������ַtxt�ļ�
	ifstream finNeg(NegSampleAdress + "NegSample.txt");         //��������ַtxt�ļ�

	if (!finPos) {
		cout << "������txt�ļ���ȡʧ��" << endl;
		return;
	}
	if (!finNeg) {
		cout << "������txt�ļ���ȡʧ��" << endl;
		return;
	}

	Mat sampleFeatureMat; // ����ѵ������������������ɵľ��������������������ĸ���������ΪHOG������ά��  
	Mat sampleLabelMat;   // ����ѵ����������������������������������ĸ����� ����Ϊ1�� 1��ʾ��Ŀ�꣬-1��ʾ��Ŀ��  

						  //---------------------------�����ȡ������ͼƬ������HOG������-------------
	for (int num = 0; num < SAMPLE_TRUE && getline(finPos, ImgName); num++) //getline(finPos, ImgName) ���ļ�finPos�ж�ȡͼ�������ImgName
	{
		system("cls");
		cout << endl << "����������: " << ImgName << endl;
		ImgName = PosSampleAdress + ImgName;
		Mat src = imread(ImgName);

		vector<float> descriptors; //������vector���������飩�����ڴ��HOG������
		hog.compute(src, descriptors, Size(8, 8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)

		if (0 == num) //�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ�������������� 
		{
			DescriptorDim = descriptors.size(); //HOG�����ӵ�ά��   
												//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat  
			sampleFeatureMat = Mat::zeros(SAMPLE_TRUE + SAMPLE_FAIL + SAMPLE_FALSE + SAMPLE_HARD, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1   
			sampleLabelMat = Mat::zeros(SAMPLE_TRUE + SAMPLE_FAIL + SAMPLE_FALSE + SAMPLE_HARD, 1, CV_32SC1);//sampleLabelMat���������ͱ���Ϊ�з���

			//������
		}


		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];

		sampleLabelMat.at<int>(num, 0) = 1;  //������ǩ���� ֵΪ1
	}

	if (SAMPLE_FAIL > 0)
	{
		cout << endl << "�����ڲ��Լ���δ������⵽������: " << endl;
		ifstream finAug("DATA/AugPosImgList.txt");
		if (!finAug) {
			cout << "Aug positive txt�ļ���ȡʧ��" << endl;
			return;
		}

		for (int num = 0; num < SAMPLE_FAIL && getline(finAug, ImgName); ++num)
		{
			ImgName = "DATA/INRIAPerson/AugPos/" + ImgName;
			Mat src = imread(ImgName);
			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8, 8));
			for (int i = 0; i < DescriptorDim; ++i)
				sampleFeatureMat.at<float>(num + SAMPLE_TRUE, i) = descriptors[i];
			sampleLabelMat.at<int>(num + SAMPLE_TRUE, 0) = 1;
		}
	}


	for (int num = 0; num < SAMPLE_FALSE && getline(finNeg, ImgName); num++)
	{
		system("cls");
		cout << "������ͼƬ����: " << ImgName << endl;
		ImgName = NegSampleAdress + ImgName;
		Mat src = imread(ImgName);

		vector<float> descriptors;
		hog.compute(src, descriptors, Size(8, 8));

		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num + SAMPLE_TRUE, i) = descriptors[i];

		sampleLabelMat.at<int>(num + SAMPLE_TRUE + SAMPLE_FAIL, 0) = -1;
	}


	if (SAMPLE_HARD > 0)
	{
		ifstream finHardExample(HardSampleAdress + "HardSampleAdressTxt.txt");
		if (!finHardExample) {
			cout << "������txt�ļ���ȡʧ��" << endl;
			return;
		}

		for (int num = 0; num < SAMPLE_HARD && getline(finHardExample, ImgName); num++)
		{
			system("cls");
			cout << endl << "����������ͼƬ: " << ImgName << endl;
			ImgName = HardSampleAdress + ImgName;
			Mat src = imread(ImgName);

			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8, 8));

			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + SAMPLE_TRUE + SAMPLE_FALSE, i) = descriptors[i];
			sampleLabelMat.at<int>(num + SAMPLE_TRUE + SAMPLE_FAIL + SAMPLE_FALSE, 0) = -1;
		}
	}

	cout << endl << "       ��ʼѵ��..." << endl;
	svm->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);
	//  svm->trainAuto(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat,10);


	svm->save("SVM_HOG.xml");
	cout << "       ѵ����ϣ�XML�ļ��ѱ���" << endl;
}