#ifndef DAOZHA_INCLUDE_DETECTION_H_
#define DAOZHA_INCLUDE_DETECTION_H_
#include "logger.h"


class Detection{
    public:
        void loadEngine(std::string path);
        void doInference(cv::Mat & org_img);
        cv::Mat preprocessImg(cv::Mat& img, int input_w, int input_h) ;
        std::vector<Object> decode();
        void qsort_descent_inplace(std::vector<Object>&faceobjects,int left, int right);
        void qsort_descent_inplace(std::vector<Object>&faceobjects);
        float intersection_area(Object & a,Object&b);
        void nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold);

        Logger logger_;

        std::unique_ptr<IRuntime> runtime_;
        std::unique_ptr<ICudaEngine> engine_;
        std::unique_ptr<IExecutionContext> context_;


        int BATCH_SIZE_ =  1;
        int INPUT_H_    =  640;
        int INPUT_W_    =  640;
        int CHANNELS_   =  3;
        int CLASSES_    =  13;
        int NUM_        =  25200;
        char * images_  =  "images";
        char * output0_ =  "output0";

        int org_img_rows_;
        int org_img_cols_;



};
#endif

