
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
��С���ܵĲ���ֵ��ͨ������£������㣬����ʱ�����㷨���ܻ�ı�ͼ���������������Ҫ����Ӧ�ĵ�����
numDisparities
�������ȥ��С���졣��ֵ���Ǵ����㡣�ڵ�ǰ��ʵ���У��ò���������Ա�16������
BLOCKSIZE
ƥ��Ŀ��С����������> = 1��������ͨ������£���Ӧ����3..11�ķ�Χ�ڡ�
P1
�����Ӳ�ƽ���ȵĵ�һ�������������ġ�
P2
�ڶ������������Ӳ�ƽ���ȡ�ֵԽ�󣬲���Խƽ����P1����������֮����Ӳ�仯�ӻ��1�ĳͷ���P2����������֮����Ӳ�仯����1�ĳͷ������㷨��ҪP2> P1����μ�stereo_match.cppʾ����������ʾ��һЩ�൱�õ�P1��P2ֵ���ֱ�Ϊ8 * number_of_image_channels * SADWindowSize * SADWindowSize��32 * number_of_image_channels * SADWindowSize * SADWindowSize����
disp12MaxDiff
�����Ӳ���������������죨����������Ϊ��λ������������Ϊ����ֵ�Խ��ü�顣
preFilterCap
Ԥ�˲�ͼ�����صĽض�ֵ�����㷨���ȼ���ÿ�����ص�x��������ͨ��[-preFilterCap��preFilterCap]���������ֵ�����ֵ���ݸ�Birchfield-Tomasi���سɱ�������
uniquenessRatio
��ѣ���С������ɱ�����ֵӦ�á�Ӯ���ڶ������ֵ�Կ����ҵ���ƥ����ȷ�İٷֱȱ�֤��ͨ����5-15��Χ�ڵ�ֵ���㹻�ˡ�
speckleWindowSize
ƽ���Ӳ���������ߴ磬�Կ����������ߵ����Ч����������Ϊ0�ɽ��ðߵ���ˡ����򣬽���������50-200�ķ�Χ�ڡ�
speckleRange
ÿ����������ڵ�����Ӳ�仯����������ߵ���ˣ�����������Ϊ��ֵ����������ʽ����16.ͨ����1��2���㹻���ˡ�
mode
��������ΪStereoSGBM :: MODE_HH������ȫ�ߴ�˫ͨ����̬����㷨����������O��W * H * numDisparities���ֽڣ����640x480�������ܴ󣬶���HD�ߴ��ͼƬ�ܴ�Ĭ������£���������Ϊfalse��

*/
