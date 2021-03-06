#include "stdafx.h"
using namespace std;
using namespace cv;

int main()
{
	FILE *cor_txt;
	fopen_s(&cor_txt, "D:\\FDU\\Tracking\\XY.txt", "r");
	double coor_ori[3][24];
	/*notice:Size(cols,rows)*/
	Mat coor_sec(Size(9, 12), CV_64FC1, Scalar(0)),
		coor_thi(Size(9, 12), CV_64FC1, Scalar(0)),
		ker_matrix = (Mat_<double>(3, 3) << 2204.93608, 0, 1019.5, 0, 2226.75866, 1023.5, 0, 0, 1),
		ker_t = ker_matrix.inv();
	transpose(ker_matrix, ker_t);

	vector<vector<Point2d>>P_essen;
	vector<Point2d>P_ori;
	//get data
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 12; ++j)
		{
			fscanf_s(cor_txt, "%lf%lf", &(coor_ori[i][j * 2]), &(coor_ori[i][1 + j * 2]));
		}
	}
	fclose(cor_txt);

	double* pxvec;
	//calculate the fundmental matrix by Least Square Method(LSM)
	for (int i = 0; i < 12; ++i)
	{
		P_ori.push_back(Point2d(coor_ori[0][2 * i], coor_ori[0][2 * i + 1]));
	}
	P_essen.push_back(P_ori);
	for (int k = 0; k < 2; ++k)
	{
		vector<Point2d>temp_p;
		for (int i = 0; i < 12; ++i)
		{
			temp_p.push_back(Point2d{ coor_ori[k + 1][2 * i],coor_ori[k + 1][2 * i + 1] });

			if (k)
				pxvec = coor_thi.ptr<double>(i);
			else
				pxvec = coor_sec.ptr<double>(i);
			pxvec[0] = coor_ori[0][2 * i] * coor_ori[k + 1][2 * i];
			pxvec[1] = coor_ori[k + 1][2 * i] * coor_ori[0][2 * i + 1];
			pxvec[2] = coor_ori[k + 1][2 * i];
			pxvec[3] = coor_ori[k + 1][2 * i + 1] * coor_ori[0][2 * i];
			pxvec[4] = coor_ori[k + 1][2 * i + 1] * coor_ori[0][2 * i + 1];
			pxvec[5] = coor_ori[k + 1][2 * i + 1];
			pxvec[6] = coor_ori[0][2 * i];
			pxvec[7] = coor_ori[0][2 * i + 1];
			pxvec[8] = 1.0;
		}
		P_essen.push_back(temp_p);
	}
	//use Singular Value Decompposition(SVD) to solve LSM
	Mat U_sec, W_sec, V_sec,
		U_thi, W_thi, V_thi,
		fund_matrix(Size(3, 3), CV_64FC1, Scalar(0));
	SVD::compute(coor_sec, W_sec, U_sec, V_sec);
	SVD::compute(coor_thi, W_thi, U_thi, V_thi);

	//get fundamental matrix
	Mat ov_fund_mat = findFundamentalMat(P_essen[0], P_essen[1]);
	for (int i = 0, i_counter = 0; i < 3; ++i)
	{
		auto pxv = fund_matrix.ptr<double>(i);
		for (int j = 0; j < 3; ++j)
		{
			pxvec = V_sec.ptr<double>(i_counter);
			pxv[j] = pxvec[3];
			i_counter++;
		}
	}

	//get essential matrix
	Mat essen_matrix = ker_t * fund_matrix*ker_matrix,
		project1(Size(4, 3), CV_64FC1, Scalar(0)),
		project2(Size(4, 3), CV_64FC1, Scalar(0));
	Mat ov_essen_mat = findEssentialMat(P_essen[0], P_essen[1], ker_matrix);
	//calculate projection matrix& translation matrix
	for (int i = 0; i < 3; ++i)
	{
		pxvec = project1.ptr<double>(i);
		for (int j = 0; j < 3; ++j)
		{
			pxvec[j] = 1.0;
		}
	}
	project1 = ker_matrix * project1;
	//get rotation matrix
	Mat R2,T2,R3,T3;
	recoverPose(ov_essen_mat, P_essen[0], P_essen[1], R2, T2);

	//now use LSM & divide-and-conquer to calculate s
	double sta_s = 0., end_s = 10.0,mid_s,dists = 10;
	while (fabs(sta_s-end_s) > 1e-3) 
	{
		mid_s = (sta_s+end_s) / 2.0;
		for (int i = 0; i < 3; ++i)
		{
			pxvec = project2.ptr<double>(i);
			auto px = R2.ptr<double>(i);
			auto py = T2.ptr<double>(i);
			for (int j = 0; j < 3; ++j)
			{
				pxvec[j] = px[j];
			}
			pxvec[3] = mid_s*py[0];
		}
		project2 = ker_matrix * project2;

		Mat pa_lef = (Mat_<double>(3, 3) << 0, -1, coor_ori[0][3], 1, 0, -coor_ori[0][2], -coor_ori[0][3], coor_ori[0][2], 0),
			pa_rig = (Mat_<double>(3, 3) << 0, -1, coor_ori[0][9], 1, 0, -coor_ori[0][8], -coor_ori[0][9], coor_ori[0][8], 0),
			pb_lef = (Mat_<double>(3, 3) << 0, -1, coor_ori[1][3], 1, 0, -coor_ori[1][2], -coor_ori[1][3], coor_ori[1][2], 0),
			pb_rig = (Mat_<double>(3, 3) << 0, -1, coor_ori[1][9], 1, 0, -coor_ori[1][8], -coor_ori[1][9], coor_ori[1][8], 0);
		pa_lef *= project1, pa_rig *= project2;
		pb_lef *= project1, pb_rig *= project2;
		Mat U_a, W_a, V_a,
			U_b, W_b, V_b,
			pa_mat, pb_mat;
		vconcat(pa_lef, pa_rig, pa_mat);
		vconcat(pb_lef, pb_rig, pb_mat);
		SVD::compute(pa_mat, W_a, U_a, V_a);
		SVD::compute(pb_mat, W_b, U_b, V_b);
		//get the 3D-coordinate of point a and point b
		int rank_fir = 0, rank_sec = 0;
		for (int i_r = 0; i_r < 4; ++i_r)
		{
			if (fabs(W_a.at<double>(i_r, 0)) < 1e-6 || i_r == 3)
			{
				rank_fir = i_r;
				break;
			}
		}
		for (int i_r = 0; i_r < 4; ++i_r)
		{
			if (fabs(W_b.at<double>(i_r, 0)) < 1e-6 || i_r == 3)
			{
				rank_sec = i_r;
				break;
			}
		}
		dists = 0;
		//normalize the coordinate of point a and b when you calculate the distance
		double norm_a = fabs(V_a.at<double>(3, rank_fir)), 
			   norm_b = fabs(V_b.at<double>(3, rank_sec));
		for (int i_d = 0; i_d < 3; ++i_d)
		{
			double dd = (V_a.at<double>(i_d, rank_fir)/norm_a-V_b.at<double>(i_d,rank_sec)/norm_b);
			dd *= dd;
			dists += dd;
		}
		if (fabs(1.5129 - dists) < 1e-3)
			break;
		else if (1.5129 > dists)
			sta_s = mid_s;
		else
			end_s = mid_s;
	}
	//calculate the modified project matrix
	Mat project2_new,res_3d;
	hconcat(R2,mid_s*T2,project2_new);
	triangulatePoints(project1, project2_new, P_essen[0], P_essen[1], res_3d);
	double norm_res;
	cout << res_3d << endl;
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			norm_res = res_3d.at<double>(i, j * 4 + 3);
			cout << norm_res << endl;
			for (int j_p = 0; j_p < 3; ++j_p)
				cout <<setw(10)<< res_3d.at<double>(i, j * 4 + j_p) / norm_res;
			cout << endl;
		}
	}
	return 0;
}

