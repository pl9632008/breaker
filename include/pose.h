#ifndef DAOZHA_INCLUDE_POSE_H_
#define DAOZHA_INCLUDE_POSE_H_
#include "logger.h"

class Pose{
    public:
        void loadEngine(std::string path);
        void doInference(cv::Mat & org_img,Object & obj );
        cv::Mat preprocessImg(cv::Mat& img, int input_w, int input_h);
        void decode(cv::Mat& org_img, Object & obj);
        void drawObjects(cv::Mat& image, std::vector<Object>&objects,cv::VideoWriter&writer);
        float calAngle(cv::Point2f  p1 , cv::Point2f p2);

        Logger logger_;

        std::unique_ptr<IRuntime> runtime4_;
        std::unique_ptr<ICudaEngine> engine4_;
        std::unique_ptr<IExecutionContext> context4_;

        std::unique_ptr<IRuntime> runtime7_;
        std::unique_ptr<ICudaEngine> engine7_;
        std::unique_ptr<IExecutionContext> context7_;
                
        std::unique_ptr<IRuntime> runtime11_;
        std::unique_ptr<ICudaEngine> engine11_;
        std::unique_ptr<IExecutionContext> context11_;

        int BATCH_SIZE_ =  1;
        int INPUT_H_    =  256;
        int INPUT_W_    =  192;
        int CHANNELS_   =  3;

        int CLASSESGW7_ =  4;
        int CLASSESGW4_ =  4;
        int CLASSESGW11_=  3;

        int OUTPUT_NUM_    =  384;
        int OUTPUT689_NUM_ =  512;

        char * input_     = "input";
        char * output_    = "output";
        char * output689_ = "689";

        int org_img_rows_;
        int org_img_cols_;


        char * class_names[8] = {"GW4_kai","GW4_he","GW7_kai","GW7_he",
                                     "GW10_kai","GW10_he","GW11_kai","GW11_he"
        };



};

#endif