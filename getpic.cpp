
#include <iostream>
#include <opencv2/opencv.hpp>



using namespace std;

using namespace cv;





int main()

{



    cv::VideoCapture capl(0);

    cv::VideoCapture capr(2);




    int i = 0;




    cv::Mat src_imgl;

    cv::Mat src_imgr;



    char filename_l[15];

    char filename_r[15];

    while (capl.read(src_imgl) && capr.read(src_imgr))

    {







        cv::imshow("src_imgl", src_imgl);

        cv::imshow("src_imgr", src_imgr);



        char c = cv::waitKey(1);

        if (c == ' ') //按空格采集图像

        {

            sprintf_s(filename_l, "left%d.jpg", i);

            imwrite(filename_l, src_imgl);

            sprintf_s(filename_r, "right%d.jpg", i++);

            imwrite(filename_r, src_imgr);

        }

        if (c == 'q' || c == 'Q') // 按q退出

        {

            break;

        }





    }



    return 0;

}
