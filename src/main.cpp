#include "detection.h"
#include "pose.h"

int main(int argc, char * argv[]){
    Detection mydect;
    Pose mypose;

    mydect.loadEngine("/wangjiadong/yolov5/runs/train/exp45/weights/yolo.engine");
    mypose.loadEngine("/wangjiadong/mmpose-main/GW4deploy/GW4.engine");
    mypose.loadEngine("/wangjiadong/mmpose-main/GW7deploy/GW7.engine");
    mypose.loadEngine("/wangjiadong/mmpose-main/GW11deploy/GW11.engine");

    // cv::VideoCapture cap(argv[1]);
    // cv::VideoWriter writer("../video/out.avi",cv::VideoWriter::fourcc('M','J','P','G'), cap.get(cv::CAP_PROP_FPS), cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH),cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

    // while (cap.isOpened()){
    //     cv::Mat org_img;
    //     cap>>org_img;
    //     if(org_img.empty()){
    //         break;
    //     }
    //     mydect.org_img_rows_ = org_img.rows;
    //     mydect.org_img_cols_ = org_img.cols;
    //     mydect.doInference(org_img);
    //     std::vector<Object>objects = mydect.decode();
        
    //     for(int i = 0 ; i < objects.size(); i++){
    //         Object& obj = objects[i];    
    //         cv::Mat img = org_img(obj.rect);
    //         mypose.doInference(img,obj);
    //         mypose.decode(img,obj);
    //     }
    //     mypose.drawObjects(org_img,objects,writer);
    // }
       
       cv::VideoWriter writer;

        cv::Mat org_img = cv::imread(argv[1]);
   
        mydect.org_img_rows_ = org_img.rows;
        mydect.org_img_cols_ = org_img.cols;
        mydect.doInference(org_img);
        std::vector<Object>objects = mydect.decode();
        
        for(int i = 0 ; i < objects.size(); i++){
            Object &obj = objects[i];    
            cv::Mat img = org_img(obj.rect);
            mypose.doInference(img,obj);
            mypose.decode(img,obj);
        }
        
        mypose.drawObjects(org_img,objects,writer);

}