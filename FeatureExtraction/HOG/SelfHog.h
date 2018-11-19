#pragma once
#ifndef SELFHOG_H
#define SELFHOG_H

#include <fstream>  
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/opencv.hpp>
#include <stdlib.h> //srand()和rand()函数  
#include <string>
#include <time.h> //time()函数  

#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define SAMPLE_FAIL 0
#define SAMPLE_FALSE 80
#define SAMPLE_HARD 0
#define SAMPLE_NUM 10
#define SAMPLE_TRUE 7


void train_SVM_HOG();
void SVM_HOG_detect();

#endif
