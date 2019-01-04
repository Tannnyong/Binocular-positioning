/******************************/
/*        立体匹配和测距        */
/******************************/

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
//Ptr<StereoBM> bm = StereoBM::create(16, 9);

StereoBM bm;
CvStereoBMState *BMState = cvCreateStereoBMState();


/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 445.10883, 0, 313.59893,
    0, 444.49320, 230.19410,
0, 0, 1);
//对应matlab里的左相机标定矩阵
Mat distCoeffL = (Mat_<double>(5, 1) << 0.06798, - 0.08190, - 0.00029,   0.00336,  0.00000);
//对应Matlab所得左i相机畸变参数

Mat cameraMatrixR = (Mat_<double>(3, 3) << 448.41168, 0, 339.85855,
    0, 448.11352, 261.12292,
    0, 0, 1);
//对应matlab里的右相机标定矩阵

Mat distCoeffR = (Mat_<double>(5, 1) << 0.07852, - 0.08038,   0.00415,   0.00047,  0.00000);
//对应Matlab所得右相机畸变参数

Mat T = (Mat_<double>(3, 1) << -0.09861,   0.00216,  0.00255);//T平移向量
//对应Matlab所得T参数
Mat rec = (Mat_<double>(3, 1) << 0.01650, 0.00849, -0.01861);//rec旋转向量，对应matlab om参数
Mat R;//R 旋转矩阵


/*****立体匹配*****/
void stereo_match(int, void*)
{



    cv::StereoSGBM sgbm;
    int SADWindowSize = 12;
    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
//    int cn = img1->nChannels;
    int numberOfDisparities = 144;
    sgbm.P1 = 8 * 3*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32 * 3*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 300;
    sgbm.speckleRange = 32;
    sgbm.disp12MaxDiff = 1;
    Mat disp, disp8;
    int64 t = getTickCount();
    sgbm(rectifyImageL, rectifyImageR, disp);
    t = getTickCount() - t;
    cout << "Time elapsed:" << t * 1000 / getTickFrequency() << endl;
    disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
    reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    imshow("disparity", disp8);


    //bm.state->preFilterCap = 31;
    //bm.state->SADWindowSize = 19;
    //bm.state->minDisparity = 0;
    //bm.state->numberOfDisparities = 96;
    //bm.state->textureThreshold = 10;
    //bm.state->uniquenessRatio = 25;
    //bm.state->speckleWindowSize = 100;
    //bm.state->speckleRange = 32;
    //bm.state->disp12MaxDiff = -1;
    //Mat disp, disp8;
    //bm(rectifyImageL, rectifyImageR, disp);
    //disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    //reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    //xyz = xyz * 16;
    //imshow("disparity", disp8);



    //bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
    //bm->setROI1(validROIL);
    //bm->setROI2(validROIR);
    //bm->setPreFilterCap(31);
    //bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
    //bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    //bm->setTextureThreshold(10);
    //bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
    //bm->setSpeckleWindowSize(100);
    //bm->setSpeckleRange(32);
    //bm->setDisp12MaxDiff(-1);
    //Mat disp, disp8;
    //bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
    //disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    //reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    //xyz = xyz * 16;
    //imshow("disparity", disp8);
}

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            break;
    }
}


/*****主函数*****/
int main()
{
    /*
    立体校正
    */
    Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    读取图片
    */
    rgbImageL = imread("left23.jpg", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
    rgbImageR = imread("right23.jpg", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

    imshow("ImageL Before Rectify", grayImageL);
    imshow("ImageR Before Rectify", grayImageR);

    /*
    经过remap之后，左右相机的图像已经共面并且行对准了
    */
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    /*
    把校正结果显示出来
    */
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    imshow("ImageL After Rectify", rgbRectifyImageL);
    imshow("ImageR After Rectify", rgbRectifyImageR);
    imwrite("rgbRectifyImageL.jpg", rgbRectifyImageL);
    imwrite("rgbRectifyImageR.jpg", rgbRectifyImageR);

    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);

    /*
    立体匹配
    */
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    stereo_match(0, 0);

    waitKey(0);
    return 0;
}

//
///******************************/
//
///*        立体匹配和测距        */
//
///******************************/
//
//#include <opencv2/opencv.hpp>  
//
//#include <highgui.h>
//
//#include <cv.h>
//
//#include <cxcore.h>
//
//#include <iostream>  
//
//using namespace std;
//
//using namespace cv;
//
//const int imageWidth = 600;                             //摄像头的分辨率  
//
//const int imageHeight = 480;
//
//Size imageSize = Size(imageWidth, imageHeight);
//
//Mat rgbImageL, grayImageL;
//
//Mat rgbImageR, grayImageR;
//
//Mat rectifyImageL, rectifyImageR;
//
//Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
//
//Rect validROIR;
//
//Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
//
//Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
//
//Mat xyz;              //三维坐标
//
//Point origin;         //鼠标按下的起始点
//
//Rect selection;      //定义矩形选框
//
//bool selectObject = false;    //是否选择对象
//
//int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
//
//StereoBM bm;
//CvStereoBMState *BMState = cvCreateStereoBMState();
//
///*
//
//事先标定好的相机的参数
//
//fx 0 cx
//
//0 fy cy
//
//0 0  1
//
//*/
//
//Mat cameraMatrixL = (Mat_<double>(3, 3) << 452.17698, 0, 272.56672,
//    0, 452.16436, 249.38020,
//    0, 0, 1);
////对应matlab里的左相机标定矩阵
//Mat distCoeffL = (Mat_<double>(5, 1) << -0.41728,   0.19365, - 0.00059,   0.00089,  0.00000);
////对应Matlab所得左i相机畸变参数
//
//Mat cameraMatrixR = (Mat_<double>(3, 3) << 448.17086, 0, 308.48848,
//    0, 447.93316, 268.71773,
//    0, 0, 1);
////对应matlab里的右相机标定矩阵
//
//Mat distCoeffR = (Mat_<double>(5, 1) << -0.40778,   0.15538, - 0.00073,   0.00107,  0.00000);
////对应Matlab所得右相机畸变参数
//
//Mat T = (Mat_<double>(3, 1) << 0.09791, - 0.00065,  0.00112);//T平移向量
////对应Matlab所得T参数
//Mat rec = (Mat_<double>(3, 1) << -0.00311, - 0.00575, - 0.02303);//rec旋转向量，对应matlab om参数
//Mat R;//R 旋转矩阵
//
///*****立体匹配*****/
//
//void stereo_match(int, void*)
//
//{
//
//    //int SADWindowSize = 15;
//    //BMState->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
//    //BMState->minDisparity = 0;
//    //BMState->numberOfDisparities = 32;
//    //BMState->textureThreshold = 10;
//    //BMState->uniquenessRatio = 15;
//    //BMState->speckleWindowSize = 100;
//    //BMState->speckleRange = 32;
//    //BMState->disp12MaxDiff = 1;
//
//    //cvFindStereoCorrespondenceBM(left, right, left_disp_, BMState);
//    //cvNormalize(left_disp_, left_vdisp, 0, 256, CV_MINMAX);
//
//        
//         int SADWindowSize = 15;
//        bm.state->preFilterCap = 31;
//        bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
//        bm.state->minDisparity = 0;
//        bm.state->numberOfDisparities = 32;
//        bm.state->textureThreshold = 10;
//        bm.state->uniquenessRatio = 25;
//        bm.state->speckleWindowSize = 100;
//        bm.state->speckleRange = 32;
//        bm.state->disp12MaxDiff = 1;
//    
//        Mat disp, disp8;
//        bm(rectifyImageL, rectifyImageR, disp);
//        disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
//        reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
//        xyz = xyz * 16;
//        imshow("disparity", disp8);
//
//}




//#include <opencv2\opencv.hpp>  
//
//#include<iostream>
//
//#include <fstream>
//
//
//
//using namespace cv;
//
//using namespace std;
//
////f is 1383.144277、B is 0.21087920785  Z = f * B /d.  f * B = 291.67636947602097445
//
//void mouseHandler(int event, int x, int y, int flags, void* param);
//
//CvMat *disp;  //disparity map
//
//bool left_mouse = false;
//
//
//
//void main()
//
//{
//
//
//
//
//
//    IplImage* imageLeft;
//
//    IplImage* imageRight;
//
//    imageLeft = cvLoadImage("image001.pgm"); //读入左侧相机的图像，3通道的
//
//    imageRight = cvLoadImage("image010.pgm");//读入右侧相机的图像  
//
//    //int d = imageLeft->depth;
//
//    int width = imageLeft->width;//图片宽度
//
//    int height = imageLeft->height;//图片高度
//
//
//
//    //这里是根据图像位数进行灰度值调整，不是必须的，可根据实际情况自行删减
//
//    cvScale(imageLeft, imageLeft, 16, 0);
//
//    cvScale(imageRight, imageRight, 16, 0);
//
//    //初始化相机的标定数据，由于不同的相机标定参数不同，所以这里用XX代替
//
//    double M1[3][3] = { xx.x00000, xx.000000, xx.xxx000, xx.000000, xx.xx0000, xx.xxx000, xx.000000, xx.000000, xx.000000 };
//
//    double M2[3][3] = { xx.x00000, xx.000000, xx.xxx000, xx.000000, xx.xx0000, xx.xxx000, xx.000000, xx.000000, xx.000000 };
//
//    double D1[5] = { x.xxx, x.xxx0, xx.xxx, x.xxx, x.xxx };
//
//    double D2[5] = { x.xxx, x.xxx0, xx.xxx, x.xxx, x.xxx };
//
//    CvMat _M1calib = cvMat(3, 3, CV_64F, M1);
//
//    CvMat _M2calib = cvMat(3, 3, CV_64F, M2);
//
//    CvMat _D1 = cvMat(1, 5, CV_64F, D1);
//
//    CvMat _D2 = cvMat(1, 5, CV_64F, D2);
//
//
//
//    double R1[3][3] = { xx.x00000, xx.000000, xx.xxx000, xx.000000, xx.xx0000, xx.xxx000, xx.000000, xx.000000, xx.000000 };
//
//    double R2[3][3] = { xx.x00000, xx.000000, xx.xxx000, xx.000000, xx.xx0000, xx.xxx000, xx.000000, xx.000000, xx.000000 };
//
//    double P1[3][4] = { xx.x00000, xx.000000, xx.xxx000, xx.000000, xx.xx0000, xx.xxx000, xx.000000, xx.000000, xx.000000, xx.000000, xx.000000, xx.000000 };
//
//    double P2[3][4] = { xx.x00000, xx.000000, xx.xxx000, xx.000000, xx.xx0000, xx.xxx000, xx.000000, xx.000000, xx.000000, xx.000000, xx.000000, xx.000000 };
//
//    CvMat _R1 = cvMat(3, 3, CV_64F, R1);
//
//    CvMat _R2 = cvMat(3, 3, CV_64F, R2);
//
//    CvMat _P1 = cvMat(3, 4, CV_64F, P1);
//
//    CvMat _P2 = cvMat(3, 4, CV_64F, P2);
//
//
//
//    CvMat* mx1calib = cvCreateMat(height, width, CV_32F);
//
//    CvMat* my1calib = cvCreateMat(height, width, CV_32F);
//
//    CvMat* mx2calib = cvCreateMat(height, width, CV_32F);
//
//    CvMat* my2calib = cvCreateMat(height, width, CV_32F);
//
//    //双目校正，分别得到左右两个相机的X坐标重映射矩阵和Y坐标重映射矩阵
//
//    cvInitUndistortRectifyMap(&_M1calib, &_D1, &_R1, &_P1, mx1calib, my1calib);
//
//    cvInitUndistortRectifyMap(&_M2calib, &_D2, &_R2, &_P2, mx2calib, my2calib);
//
//
//
//
//
//    CvMat	*img1r,		//rectified left image
//
//        *img2r,		//rectified right image		
//
//        *vdisp,		//scaled disparity map for viewing
//
//        *pair,
//
//        *depthM;
//
//    img1r = cvCreateMat(height, width, CV_8U);		//rectified left image
//
//    img2r = cvCreateMat(height, width, CV_8U);		//rectified right image
//
//    disp = cvCreateMat(height, width, CV_16S);		//disparity map
//
//    vdisp = cvCreateMat(height, width, CV_8U);
//
//    CvStereoBMState *BMState = cvCreateStereoBMState();
//
//    assert(BMState != 0);
//
//    //BMState->preFilterSize = 63;//stereoPreFilterSize; 
//
//    //BMState->preFilterCap = 63;//stereoPreFilterCap;
//
//    BMState->SADWindowSize = 15;// stereoDispWindowSize; //33
//
//    BMState->minDisparity = 0;
//
//    BMState->numberOfDisparities = 48;//stereoNumDisparities; //48
//
//    BMState->textureThreshold = 20;//stereoDispTextureThreshold; //20
//
//    BMState->uniquenessRatio = 15;///stereoDispUniquenessRatio;//15
//
//    BMState->speckleWindowSize = 200;
//
//    BMState->speckleRange = 32;
//
//    BMState->disp12MaxDiff = 2;
//
//
//
//    IplImage	*img1,	//left image
//
//        *img2;
//
//
//
//    img1 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
//
//    img2 = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
//
//    cvCvtColor(imageLeft, img1, CV_BGR2GRAY);
//
//    cvCvtColor(imageRight, img2, CV_BGR2GRAY);
//
//
//
//    cvRemap(img1, img1r, mx1calib, my1calib);
//
//    cvRemap(img2, img2r, mx2calib, my2calib);
//
//
//
//    cvFindStereoCorrespondenceBM(img1r, img2r, disp, BMState);
//
//    cvNormalize(disp, vdisp, 0, 256, CV_MINMAX);
//
//
//
//
//
//    //cvNamedWindow( "Rectified", 1);
//
//    //cvNamedWindow( "uDisparity Map",0 );
//
//    cvNamedWindow("Disparity Map", 0);
//
//    //cvShowImage("Rectified", pair);
//
//    //cvShowImage("uDisparity Map", disp);
//
//    cvShowImage("Disparity Map", vdisp);
//
//    cvSetMouseCallback("Disparity Map", mouseHandler, NULL);
//
//    cvNamedWindow("left", 0);
//
//    //cvNamedWindow("right",0);
//
//
//
//    cvShowImage("left", imageLeft);
//
//    //cvShowImage("right",imageRight);
//
//
//
//    cvWaitKey(0);
//
//
//
//    cvDestroyWindow("Rectified");
//
//    cvDestroyWindow("Disparity Map");
//
//    cvDestroyWindow("left");
//
//    //cvDestroyWindow("right");
//
//    cvReleaseImage(&imageLeft);
//
//    cvReleaseImage(&imageRight);
//
//}
//
//
//
//void mouseHandler(int event, int x, int y, int flags, void *param){
//
//
//
//    if (event == CV_EVENT_LBUTTONDOWN)
//
//    {
//
//        cout << "x:" << x << "y:" << y << endl;
//
//        //l = cvGet2D(stereoFunc->depthM, x, y);
//
//        CvScalar s = cvGet2D(disp, y, x);
//
//        double dep1 = s.val[0];
//
//        double dep = 291.67636947602097445 / dep1;
//
//        dep *= 16;
//
//        //int dep2 = cvmGet( vdisp, 1000, 500 );
//
//        //printf("dep1 = %d\n",dep1);
//
//        printf("Distance to this object is: %f m \n", dep);
//
//        left_mouse = true;
//
//
//
//    }
//
//    else if (event == CV_EVENT_LBUTTONUP)
//
//    {
//
//        left_mouse = false;
//
//    }
//
//    else if ((event == CV_EVENT_MOUSEMOVE) && (left_mouse == true))
//
//    {
//
//    }
//
//}