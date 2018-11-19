#include"SelfHog.h"
using namespace std;
using namespace cv;
using namespace cv::ml;

void train_SVM_HOG()
{

	HOGDescriptor hog(Size(IMG_WIDTH, IMG_HEIGHT), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	int DescriptorDim; //HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定  

	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	//  svm->setC(0.01); //设置惩罚参数C，默认值为1
	svm->setKernel(SVM::LINEAR); //线性核
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 3000, 1e-6)); //3000是迭代次数，1e-6是精确度。
																			//setTermCriteria是用来设置算法的终止条件， 
	//SVM训练的过程就是一个通过 迭代 方式解决约束条件下的二次优化问题，这里我们指定一个最大迭代次数和容许误
	//差，以允许算法在适当的条件下停止计算


	string ImgName;//图片的名字
	string PosSampleAdress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\true\\";
	string NegSampleAdress = "D:\\FDU\\Template\\CS\\ComputerVision\\smart-phone_dataset\\false\\";
	string HardSampleAdress = "";

	ifstream finPos(PosSampleAdress + "PosSample.txt"); //正样本地址txt文件
	ifstream finNeg(NegSampleAdress + "NegSample.txt");         //负样本地址txt文件

	if (!finPos) {
		cout << "正样本txt文件读取失败" << endl;
		return;
	}
	if (!finNeg) {
		cout << "负样本txt文件读取失败" << endl;
		return;
	}

	Mat sampleFeatureMat; // 所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数为HOG描述子维数  
	Mat sampleLabelMat;   // 所有训练样本的类别向量，行数等于所有样本的个数， 列数为1： 1表示有目标，-1表示无目标  

						  //---------------------------逐个读取正样本图片，生成HOG描述子-------------
	for (int num = 0; num < SAMPLE_TRUE && getline(finPos, ImgName); num++) //getline(finPos, ImgName) 从文件finPos中读取图像的名称ImgName
	{
		system("cls");
		cout << endl << "正样本处理: " << ImgName << endl;
		ImgName = PosSampleAdress + ImgName;
		Mat src = imread(ImgName);

		vector<float> descriptors; //浮点型vector（类似数组），用于存放HOG描述子
		hog.compute(src, descriptors, Size(8, 8));//计算HOG描述子，检测窗口移动步长(8,8)

		if (0 == num) //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵 
		{
			DescriptorDim = descriptors.size(); //HOG描述子的维数   
												//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat  
			sampleFeatureMat = Mat::zeros(SAMPLE_TRUE + SAMPLE_FAIL + SAMPLE_FALSE + SAMPLE_HARD, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1   
			sampleLabelMat = Mat::zeros(SAMPLE_TRUE + SAMPLE_FAIL + SAMPLE_FALSE + SAMPLE_HARD, 1, CV_32SC1);//sampleLabelMat的数据类型必须为有符号

			//整数型
		}


		for (int i = 0; i<DescriptorDim; i++)
			sampleFeatureMat.at<float>(num, i) = descriptors[i];

		sampleLabelMat.at<int>(num, 0) = 1;  //样本标签矩阵 值为1
	}

	if (SAMPLE_FAIL > 0)
	{
		cout << endl << "处理在测试集中未被被检测到的样本: " << endl;
		ifstream finAug("DATA/AugPosImgList.txt");
		if (!finAug) {
			cout << "Aug positive txt文件读取失败" << endl;
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
		cout << "负样本图片处理: " << ImgName << endl;
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
			cout << "难样本txt文件读取失败" << endl;
			return;
		}

		for (int num = 0; num < SAMPLE_HARD && getline(finHardExample, ImgName); num++)
		{
			system("cls");
			cout << endl << "处理难样本图片: " << ImgName << endl;
			ImgName = HardSampleAdress + ImgName;
			Mat src = imread(ImgName);

			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8, 8));

			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + SAMPLE_TRUE + SAMPLE_FALSE, i) = descriptors[i];
			sampleLabelMat.at<int>(num + SAMPLE_TRUE + SAMPLE_FAIL + SAMPLE_FALSE, 0) = -1;
		}
	}

	cout << endl << "       开始训练..." << endl;
	svm->train(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);
	//  svm->trainAuto(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat,10);


	svm->save("SVM_HOG.xml");
	cout << "       训练完毕，XML文件已保存" << endl;
}