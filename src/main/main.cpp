#include "faster_rcnn.hpp"
using namespace cv;
int main()
{
    string model_file = "/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/models/pascal_voc/VGG16/faster_rcnn_end2end/test.prototxt";
    string weights_file = "/home/xieqiang/Documents/Code/Detection/py-faster-rcnn-master/data/faster_rcnn_models/PascalAndDetrac_iter_50000.caffemodel";
    int GPUID=0;
    vector<vector<int> > ans;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    Detector det = Detector(model_file, weights_file);
    cv::Mat im = cv::imread("test1.jpg");
    ans = det.Detect(im);
    for(int i = 0;i < ans.size();++i){
        for(int j = 0;j < ans[i].size();j++){
            cout << ans[i][j] << " ";
        }
	rectangle(im,cvPoint(ans[i][0],ans[i][1]),cvPoint(ans[i][2] + ans[i][0],ans[i][3] + ans[i][1]),Scalar(0,0,255),1,1,0);
        cout << endl;
    }
    imwrite("test.jpg",im);
    return 0;
}
