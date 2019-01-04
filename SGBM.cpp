
#include <cvaux.h>
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <iostream>


using namespace std;
using namespace cv;
int main()
{

    IplImage * img1 = cvLoadImage("rgbRectifyImageL.jpg", 0);
    IplImage * img2 = cvLoadImage("rgbRectifyImageR.jpg", 0);
    cv::StereoSGBM sgbm;
    int SADWindowSize = 11;
    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
    int cn = img1->nChannels;
    int numberOfDisparities = 144;
    sgbm.P1 = 8 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.numberOfDisparities = numberOfDisparities;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 300;
    sgbm.speckleRange = 10;
    sgbm.disp12MaxDiff = 1;
    Mat disp, disp8;
    int64 t = getTickCount();
    sgbm((Mat)img1, (Mat)img2, disp);
    t = getTickCount() - t;
    cout << "Time elapsed:" << t * 1 / getTickFrequency() << endl;
    disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));

    Mat img3,img4,img5;
    threshold(disp8, img3, 160, 255, THRESH_BINARY);
    threshold(disp8, img4, 80, 255, THRESH_BINARY);
    threshold(disp8, img5, 0, 255, THRESH_BINARY);
    
    img5 = img5 - img4;
    img4 = img4 - img3;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
   // morphologyEx(disp8, disp8, MORPH_CLOSE, element);

   /* cvNamedWindow("1", 0);
    imshow("1", img3);
    cvNamedWindow("2", 0);
    imshow("2", img4);
    cvNamedWindow("3", 0);
    imshow("3", img5);*/
    namedWindow("left", 1);
    cvShowImage("left", img1);
    namedWindow("right", 1);
    cvShowImage("right", img2);
    namedWindow("disparity", 1);
    imshow("disparity", disp8);
    imwrite("sgbm_disparity.png", disp8);
    waitKey();
    cvDestroyAllWindows();
    return 0;
}



/*
minDisparity
最小可能的差异值。通常情况下，它是零，但有时整流算法可能会改变图像，所以这个参数需要作相应的调整。
numDisparities
最大差异减去最小差异。该值总是大于零。在当前的实现中，该参数必须可以被16整除。
BLOCKSIZE
匹配的块大小。它必须是> = 1的奇数。通常情况下，它应该在3..11的范围内。
P1
控制视差平滑度的第一个参数。见下文。
P2
第二个参数控制视差平滑度。值越大，差异越平滑。P1是相邻像素之间的视差变化加或减1的惩罚。P2是相邻像素之间的视差变化超过1的惩罚。该算法需要P2> P1。请参见stereo_match.cpp示例，其中显示了一些相当好的P1和P2值（分别为8 * number_of_image_channels * SADWindowSize * SADWindowSize和32 * number_of_image_channels * SADWindowSize * SADWindowSize）。
disp12MaxDiff
左右视差检查中允许的最大差异（以整数像素为单位）。将其设置为非正值以禁用检查。
preFilterCap
预滤波图像像素的截断值。该算法首先计算每个像素的x导数，并通过[-preFilterCap，preFilterCap]间隔剪切其值。结果值传递给Birchfield-Tomasi像素成本函数。
uniquenessRatio
最佳（最小）计算成本函数值应该“赢”第二个最佳值以考虑找到的匹配正确的百分比保证金。通常，5-15范围内的值就足够了。
speckleWindowSize
平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
speckleRange
每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了。
mode
将其设置为StereoSGBM :: MODE_HH以运行全尺寸双通道动态编程算法。它将消耗O（W * H * numDisparities）字节，这对640x480立体声很大，对于HD尺寸的图片很大。默认情况下，它被设置为false。

*/
