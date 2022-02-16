//
//  ViewController.m
//  OpenCVDemo
//
//  Created by Murphy Zheng on 17/8/23.
//  Copyright © 2017年 mieasy. All rights reserved.
//
#import "UIImage+OpenCV.h"
#import "ViewController.h"
#import "faceRecognition.h"

using namespace std;
using namespace cv;

@interface ViewController ()
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@end


@implementation ViewController
RNG rng(12345);



//初始化mat
//    int sz[3] = {2,2,2};
//    Mat L(3,sz,CV_8UC3,Scalar::all(0));

//Mat::zeros,Mat::ones,Mat::eye

//输出mat
//    cout << "inputMat = " << inputMat << ";" << endl << endl;

//getTickCount() 返回cpu自某个事件以来走过的时钟周期数
//getTickFrequency() 返回cpu一秒钟所走的时钟周期数

//    double time0 = static_cast<double>(getTickCount());
//    //进行图像操作处理
//    // 灰度处理
//
//
//    time0 = ((double)getTickCount() - time0)/getTickFrequency();
//    cout<<"此方法运行时间为:"<<time0<<"秒"<<endl;


- (void)viewDidLoad {
    [super viewDidLoad];
    

//    [self roiImage];
    
//    [self laplacian];
    
//    [self sobel];
    
//    [self canny];
    
//    [self threshold];

//    [self GaussianBlur];
    
//    [self loadNetPic];
    
    [self dft];
    
//    [self cartoon];
//    // 人脸识别
//    UIImage *faceImage = [UIImage imageNamed:@"zhuxian.jpeg"];
//    self.imageView.image = [faceRecognition faceDetectForImage:faceImage];
}

- (void)connectArea{
    UIImage *image = [UIImage imageNamed:@"sample"];
    cv::Mat inputMat = [UIImage cvMatFromUIImage:image];

    cv::Mat resultyMat;

    bitwise_not(inputMat, inputMat);
    
    
    UIImage *inputMatImage = [UIImage imageWithCVMat:inputMat];

    cv::Mat greyMat,binary;
    cv::cvtColor(inputMat, greyMat, COLOR_BGR2GRAY);
    //cv::Mat greyMat = [UIImage cvMatGrayFromUIImage:image];
//    threshold(greyMat,binary,0,255,THRESH_BINARY_INV);
//    threshold(greyMat, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    UIImage *greyMatImage = [UIImage imageWithCVMat:greyMat];

    UIImage *b_srcImage = [UIImage imageWithCVMat:binary];

//    medianBlur(greyMat,greyMat,5); //去噪
//    Laplacian(greyMat, greyMat, CV_8U, 5);
    
    Mat labels = Mat::zeros(binary.size(), CV_32S);
    Mat stats, centroids;

    int num_labels = connectedComponentsWithStats(binary, labels, stats, centroids, 8, CV_32S, CCL_DEFAULT);
    vector<Vec3b> colorTable(num_labels);
    // backgound color
    colorTable[0] = Vec3b(0, 0, 0);
    for (int i = 1; i < num_labels; i++) {
        colorTable[i] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
    }

    Mat result = Mat::zeros(binary.size(), CV_8UC3);
    int w = result.cols;
    int h = result.rows;
    for (int row = 0; row < h; row++) {
        for (int col = 0; col < w; col++) {
            int label = labels.at<int>(row, col);
            result.at<Vec3b>(row, col) = colorTable[label];
        }
    }

    for (int i = 1; i < num_labels; i++) {
        // center
        int cx = centroids.at<double>(i, 0);
        int cy = centroids.at<double>(i, 1);
        // rectangle and area
        int x = stats.at<int>(i, CC_STAT_LEFT);
        int y = stats.at<int>(i, CC_STAT_TOP);
        int width = stats.at<int>(i, CC_STAT_WIDTH);
        int height = stats.at<int>(i, CC_STAT_HEIGHT);
        int area = stats.at<int>(i, CC_STAT_AREA);

        // ªÊ÷∆

        circle(result, cv::Point(cx, cy), 3, Scalar(0, 0, 255), 2, 8, 0);
        cv::Rect box(x, y, width, height);
        rectangle(result, box, Scalar(0, 255, 0), 2, 8);

    }
    
    UIImage *greyImage = [UIImage imageWithCVMat:result];




    self.imageView.image = greyImage;
}

- (void)loadNetPic{
    VideoCapture cap(0);

    while (1)
    {
        Mat cam;
        cap >> cam;//获取当前帧图像
        UIImage *s_srcImage = [UIImage imageWithCVMat:cam];

        self.imageView.image = s_srcImage;
     }
    

}

- (void)GaussianBlur{
    UIImage *image = [UIImage imageNamed:@"sample"];
    Mat src = [UIImage cvMatFromUIImage:image];
    
    Mat dst;
    GaussianBlur(src, dst, cv::Size(25,25), 0, 0);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:dst];

    self.imageView.image = s_srcImage;
}

- (void)threshold{
    UIImage *image = [UIImage imageNamed:@"sample"];
    Mat src = [UIImage cvMatFromUIImage:image];
    Mat gray,dst;
    cvtColor(src, gray, COLOR_RGBA2GRAY);
    
    threshold(gray, dst, 127, 255,0);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:dst];

    self.imageView.image = s_srcImage;
}

- (void)canny{
    UIImage *image = [UIImage imageNamed:@"sample"];
    Mat src = [UIImage cvMatFromUIImage:image];
    
    Mat src1 = src.clone();
    
    Mat dst,edge,gray;
    
    dst.create(src1.size(), src1.type());
    
    cvtColor(src1, gray, COLOR_RGBA2GRAY);
    
    blur(gray, edge, cv::Size(3,3));
    
    Canny(edge, edge, 150, 75, 3);
    
    dst = Scalar::all(0);
    
    src1.copyTo(dst,edge);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:edge];

    self.imageView.image = s_srcImage;
}

- (void)sobel{
    Mat grad_x,grad_y,abs_grad_x,abs_grad_y,dst;
    
    UIImage *image = [UIImage imageNamed:@"sample"];
    Mat src = [UIImage cvMatFromUIImage:image];
    
    Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    
    Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:dst];

    self.imageView.image = s_srcImage;
}

- (void)laplacian{
    UIImage *image = [UIImage imageNamed:@"sample"];
    Mat inputMat = [UIImage cvMatFromUIImage:image];
    
    GaussianBlur(inputMat, inputMat, cv::Size(3,3), 0);
    
    Mat grayMat,dstMat,absDstMat;
    cvtColor(inputMat, grayMat, COLOR_RGBA2GRAY);
    
    
    Laplacian(grayMat, dstMat, CV_16S,3,1,0,BORDER_DEFAULT);
    
    //计算绝对值，并将结果转换为8位
    convertScaleAbs(dstMat, absDstMat);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:absDstMat];

    self.imageView.image = s_srcImage;

}

- (void)loadBundleImage{
    NSString * path =[[NSBundle mainBundle]pathForResource:@"meinv" ofType:@"jpg"];

    const char *cString1 = [path cStringUsingEncoding:NSUTF8StringEncoding];

    Mat srcImage = imread(cString1);
    
    cvtColor(srcImage, srcImage, COLOR_BGR2RGB);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:srcImage];

    self.imageView.image = s_srcImage;
}

//离散傅里叶变换
- (void)dft{
    
    UIImage *image = [UIImage imageNamed:@"sample"];
    Mat srcImage = [UIImage cvMatFromUIImage:image];
    if(!srcImage.data) {
        return ;
    }
    
//    NSString * path =[[NSBundle mainBundle]pathForResource:@"meishi" ofType:@"jpg"];
//
//    const char *cString1 = [path cStringUsingEncoding:NSUTF8StringEncoding];
//
//    Mat srcImage = imread(cString1);
    
    cvtColor(srcImage, srcImage, COLOR_RGBA2GRAY);

    
    int m = getOptimalDFTSize( srcImage.rows );
    int n = getOptimalDFTSize( srcImage.cols );

    Mat padded;
    copyMakeBorder(srcImage, padded, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    split(complexI, planes);
    
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magnitudeImage = planes[0];

    magnitudeImage += Scalar::all(1);
    log(magnitudeImage, magnitudeImage);

    magnitudeImage = magnitudeImage(cv::Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & -2));

    int cx = magnitudeImage.cols/2;
    int cy = magnitudeImage.rows/2;
    Mat q0(magnitudeImage, cv::Rect(0, 0, cx, cy));   // ROI«¯”Úµƒ◊Û…œ
    Mat q1(magnitudeImage, cv::Rect(cx, 0, cx, cy));  // ROI«¯”Úµƒ”“…œ
    Mat q2(magnitudeImage, cv::Rect(0, cy, cx, cy));  // ROI«¯”Úµƒ◊Ûœ¬
    Mat q3(magnitudeImage, cv::Rect(cx, cy, cx, cy)); // ROI«¯”Úµƒ”“œ¬

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);
    
    printBaseData(&magnitudeImage);
    
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:magnitudeImage];

    self.imageView.image = s_srcImage;

    
}

void printBaseData(Mat *A){
    int dims = A->dims;
        int rows = A->rows;
        int cols = A->cols;
        int width = A->size().width;
        int height = A->size().height;
        int channels = A->channels();
        int depth = A->depth();
        int elemSize = A->elemSize();
        int elemSize1 = A->elemSize1();
        int step = A->step;
        int step1 = A->step1();
        int type = A->type();
        cout << "dims : " << dims << endl;
        cout << "rows : " << rows << endl;
        cout << "cols : " << cols << endl;
        cout << "width : " << width << endl;
        cout << "height : " << height << endl;
        cout << "channels : " << channels << endl;
        cout << "depth : " << depth << endl;
        cout << "elemSize : " << elemSize << endl;
        cout << "elemSize1 : " << elemSize1 << endl;
        cout << "step : " << step << endl;
        cout << "step1 : " << step1 << endl;
        cout << "type : " << type << endl;

}

- (void)roiImage{
    UIImage *image = [UIImage imageNamed:@"sample"];
    cv::Mat inputMat = [UIImage cvMatFromUIImage:image];
    
    UIImage *image1 = [UIImage imageNamed:@"xiaxia"];
    cv::Mat inputMat1 = [UIImage cvMatFromUIImage:image1];
    
    Mat imageROI = inputMat(cv::Rect(100,100,inputMat1.cols,inputMat1.rows));
    
    UIImage *image2 = [UIImage imageNamed:@"xiaxia"];
    cv::Mat inputMat2 = [UIImage cvMatFromUIImage:image2];
    
    cv::Mat greyMat;

    cv::cvtColor(inputMat2, greyMat, CV_BGRA2GRAY);

    inputMat1.copyTo(imageROI, greyMat);

    UIImage *inputMatImage = [UIImage imageWithCVMat:inputMat];
    
}


//双边滤波+边缘检测
- (void)cartoon{
    // 灰度处理
    UIImage *image = [UIImage imageNamed:@"123"];
    
    cv::Mat resultyMat;

    cv::Mat inputMat = [UIImage cvMatFromUIImage:image];
    
    
    cv::Mat greyMat;
    cv::cvtColor(inputMat, greyMat, CV_BGRA2BGR);
    //cv::Mat greyMat = [UIImage cvMatGrayFromUIImage:image];
    
    medianBlur(greyMat,greyMat,5); //去噪
    Laplacian(greyMat, greyMat, CV_8U, 5);
    
    Mat mask(greyMat.size(),CV_8U);
    threshold(greyMat,mask,120,255,THRESH_BINARY_INV);
    
    
    cv::Mat input8UC3Mat = [UIImage cvMatGrayFromUIImage:image];

    
    cv::Mat s_src;
    cv::Mat tmp;

    cv::cvtColor(inputMat, s_src, CV_BGRA2BGR);
    cv::cvtColor(inputMat, tmp, CV_BGRA2BGR);

    
    UIImage *greyImage = [UIImage imageWithCVMat:greyMat];

    
    int iterator=7;
    for(int i=0;i<iterator;i++){
        int ksize=9;
        double sigmaColor=9;
        double sigmaSpace=7;
        bilateralFilter(s_src,tmp,ksize,sigmaColor,sigmaSpace);
        bilateralFilter(tmp,s_src,ksize,sigmaColor,sigmaSpace);
    }

    cv::Mat b_src,dst;
//    add(s_src, mask, b_src);
//    dst=Mat(s_src.size(),s_src.type(),Scalar::all(0)); //初始化

    addWeighted(s_src, 0.5, mask, 0.5, 0.0, b_src);

//    b_src.copyTo(dst,mask);
    
    UIImage *s_srcImage = [UIImage imageWithCVMat:s_src];
    
    
    UIImage *b_srcImage = [UIImage imageWithCVMat:b_src];

    self.imageView.image = b_srcImage;
}

@end
