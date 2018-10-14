#include "stdafx.h"
using namespace std;
using namespace cv;

int main()
{
	FILE *cor_txt;
	fopen_s(&cor_txt,"D:\\FDU\\Tracking\\XY.txt","r");
	double coor_ori[3][24];
	Mat coor_sec(Size(9, 12), CV_64FC1, Scalar(0)),
		coor_thi(Size(9, 12), CV_64FC1, Scalar(0)),ker_t,
		ker_matrix = (Mat_<double>(3, 3) << 2204.93608, 0, 1019.5, 0, 2226.75866, 1023.5, 0, 0, 1);
	transpose(ker_matrix,ker_t);
	
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 12; ++j)
		{
			fscanf_s(cor_txt,"%lf%lf",&(coor_ori[i][j*2]),&(coor_ori[i][1+j*2]));
		}
	}
	fclose(cor_txt);

	double* pxvec;
	for (int k = 0; k < 2; ++k)
	{
		for (int i = 0; i < 12; ++i)
		{
			if (k)
				pxvec = coor_thi.ptr<double>(i);
			else
				pxvec = coor_sec.ptr<double>(i);
			pxvec[0] = coor_ori[0][2 * i] * coor_ori[k+1][2 * i];
			pxvec[1] = coor_ori[k+1][2 * i] * coor_ori[0][2 * i + 1];
			pxvec[2] = coor_ori[k+1][2 * i];
			pxvec[3] = coor_ori[k+1][2 * i + 1] * coor_ori[0][2 * i];
			pxvec[4] = coor_ori[k+1][2 * i + 1] * coor_ori[0][2 * i + 1];
			pxvec[5] = coor_ori[k+1][2 * i + 1];
			pxvec[6] = coor_ori[0][2 * i];
			pxvec[7] = coor_ori[0][2 * i + 1];
			pxvec[8] = 1.0;
		}
	}
	Mat U_sec, W_sec, V_sec,
		U_thi, W_thi, V_thi,
		fund_matrix(Size(3,3),CV_64FC1,Scalar(0));
	SVD::compute(coor_sec,W_sec,U_sec,V_sec);
	SVD::compute(coor_thi,W_thi,U_thi,V_thi);
	pxvec = V_sec.ptr<double>(8);
	for (int i = 0,i_counter = 0; i < 3; ++i)
	{
		auto pxv = fund_matrix.ptr<double>(i);
		for (int j = 0; j < 3; ++j)
		{
			pxv[j] = pxvec[i_counter++];
		}
	}
	Mat essen_matrix = ker_t * fund_matrix*ker_matrix;
	return 0;
}

