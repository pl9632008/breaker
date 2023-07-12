#include "pose.h"

static float posein[3*256*192];

static float GW4out[4*384];
static float GW4out689[4*512];

static float GW7out[4*384];
static float GW7out689[4*512];

static float GW11out[3*384];
static float GW11out689[3*512];


void Pose::loadEngine(std::string path){

        size_t size{0};
        char * trtModelStream{nullptr};

        std::ifstream file(path, std::ios::binary);
        
        if(file.good()){
            file.seekg(0,std::ios::end);
            size = file.tellg();
            file.seekg(0,std::ios::beg);
            trtModelStream = new char[size];
            file.read(trtModelStream,size);
            file.close();
        }
        auto index_start = path.find_last_of("/")+1;
        auto index_end   = path.find_last_of(".");
        std::string str = path.substr(index_start,index_end-index_start);

        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger_));
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(trtModelStream,size));
        std::unique_ptr<IExecutionContext>context(engine->createExecutionContext());

        if(str == "GW4"){

            runtime4_ = std::move(runtime);
            engine4_  = std::move(engine);
            context4_ = std::move(context);
        }else if(str == "GW7"){
   
            runtime7_ = std::move(runtime);
            engine7_  = std::move(engine);
            context7_ = std::move(context);

        }else if(str == "GW11"){

            runtime11_ = std::move(runtime);
            engine11_  = std::move(engine);
            context11_ = std::move(context);

        }

        delete[] trtModelStream;
}

cv::Mat Pose::preprocessImg(cv::Mat& img, int input_w, int input_h){
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows))); 
    return out;
}


/**
 * @brief TensorRT Inference
 * @param org_img  original image before padding and resizing
 * @param label category per instance
 * 0,1 GW4
 * 2,3 GW7
 * 4,5 GW10
 * 6,7 GW11
 * @return keypoint per instance  
**/
void Pose::doInference(cv::Mat & org_img, Object & obj ){

    int label = obj.label;

    cv::Mat pr_img = preprocessImg(org_img,INPUT_W_,INPUT_H_);
        
    for(int i = 0 ; i < INPUT_W_*INPUT_H_;i++){
        posein[i] = pr_img.at<cv::Vec3b>(i)[2]/255.0;
        posein[i+INPUT_W_*INPUT_H_] = pr_img.at<cv::Vec3b>(i)[1]/255.0;
        posein[i+2*INPUT_W_*INPUT_H_]=pr_img.at<cv::Vec3b>(i)[0]/255.0;
    }

    if(label==0 || label ==1){

        int32_t input_index = engine4_->getBindingIndex(input_);
        int32_t output_index = engine4_->getBindingIndex(output_);
        int32_t output689_index = engine4_->getBindingIndex(output689_);

        void * buffers[3];
        cudaMalloc(&buffers[input_index],BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float));
        cudaMalloc(&buffers[output_index],BATCH_SIZE_*OUTPUT_NUM_*CLASSESGW4_*sizeof(float));
        cudaMalloc(&buffers[output689_index],BATCH_SIZE_*OUTPUT689_NUM_*CLASSESGW4_*sizeof(float));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[input_index],posein,BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float),cudaMemcpyHostToDevice,stream);
        context4_->enqueueV2(buffers,stream, nullptr);
        cudaMemcpyAsync(GW4out,buffers[output_index],BATCH_SIZE_*OUTPUT_NUM_*CLASSESGW4_*sizeof(float),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(GW4out689,buffers[output689_index],BATCH_SIZE_*OUTPUT689_NUM_*CLASSESGW4_*sizeof(float),cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[input_index]);
        cudaFree(buffers[output_index]);
        cudaFree(buffers[output689_index]);

    }else if(label ==2 || label==3){

        int32_t input_index = engine7_->getBindingIndex(input_);
        int32_t output_index = engine7_->getBindingIndex(output_);
        int32_t output689_index = engine7_->getBindingIndex(output689_);

        void * buffers[3];
        cudaMalloc(&buffers[input_index],BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float));
        cudaMalloc(&buffers[output_index],BATCH_SIZE_*OUTPUT_NUM_*CLASSESGW7_*sizeof(float));
        cudaMalloc(&buffers[output689_index],BATCH_SIZE_*OUTPUT689_NUM_*CLASSESGW7_*sizeof(float));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[input_index],posein,BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float),cudaMemcpyHostToDevice,stream);
        context7_->enqueueV2(buffers,stream, nullptr);
        cudaMemcpyAsync(GW7out,buffers[output_index],BATCH_SIZE_*OUTPUT_NUM_*CLASSESGW7_*sizeof(float),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(GW7out689,buffers[output689_index],BATCH_SIZE_*OUTPUT689_NUM_*CLASSESGW7_*sizeof(float),cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[input_index]);
        cudaFree(buffers[output_index]);
        cudaFree(buffers[output689_index]);

    }else if(label == 6 || label == 7){
        
        int32_t input_index = engine11_->getBindingIndex(input_);
        int32_t output_index = engine11_->getBindingIndex(output_);
        int32_t output689_index = engine11_->getBindingIndex(output689_);

        void * buffers[3];
        cudaMalloc(&buffers[input_index],BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float));
        cudaMalloc(&buffers[output_index],BATCH_SIZE_*OUTPUT_NUM_*CLASSESGW11_*sizeof(float));
        cudaMalloc(&buffers[output689_index],BATCH_SIZE_*OUTPUT689_NUM_*CLASSESGW11_*sizeof(float));

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[input_index],posein,BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float),cudaMemcpyHostToDevice,stream);
        context11_->enqueueV2(buffers,stream, nullptr);
        cudaMemcpyAsync(GW11out,buffers[output_index],BATCH_SIZE_*OUTPUT_NUM_*CLASSESGW11_*sizeof(float),cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(GW11out689,buffers[output689_index],BATCH_SIZE_*OUTPUT689_NUM_*CLASSESGW11_*sizeof(float),cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[input_index]);
        cudaFree(buffers[output_index]);
        cudaFree(buffers[output689_index]);

    }

}

void Pose::decode(cv::Mat& img,Object & obj){
        
        int label = obj.label;
        
        float r_w = INPUT_W_/(img.cols*1.0);
        float r_h = INPUT_H_/(img.rows*1.0);

        std::vector<KeyPoints> keypoints;

        if(label==0 || label ==1){
            for(int c = 0 ; c < CLASSESGW4_ ; c++){
                auto x_index = std::max_element(GW4out+c*OUTPUT_NUM_, GW4out+(c+1)*OUTPUT_NUM_) - (GW4out+c*OUTPUT_NUM_);
                auto y_index = std::max_element(GW4out689+c*OUTPUT689_NUM_, GW4out689+(c+1)*OUTPUT689_NUM_) - (GW4out689+c*OUTPUT689_NUM_);
                auto prob = std::max(  *std::max_element(GW4out+c* OUTPUT_NUM_ , GW4out+(c+1)* OUTPUT_NUM_ ) ,  *std::max_element(GW4out689+c*OUTPUT689_NUM_, GW4out689+(c+1)*OUTPUT689_NUM_)    );
                KeyPoints keypoint;
                x_index/=2;
                y_index/=2;

                if(r_h>r_w){
                    x_index = x_index/r_w;
                    y_index = (y_index - ( INPUT_H_ - r_w * img.rows )/2 )/r_w; 
                }else{

                    x_index = (x_index - ( INPUT_W_ - r_h * img.cols )/2 )/r_h;
                    y_index = y_index/r_h;

                }      
                keypoint.p = cv::Point2f(x_index +obj.rect.x  ,y_index + obj.rect.y);
                keypoint.prob = prob;
                keypoints.push_back(keypoint);

            }

        }else if(label==2 || label==3){
            for(int c = 0 ; c < CLASSESGW7_ ; c++){
                auto x_index = std::max_element(GW7out+c*OUTPUT_NUM_, GW7out+(c+1)*OUTPUT_NUM_) - (GW7out+c*OUTPUT_NUM_);
                auto y_index = std::max_element(GW7out689+c*OUTPUT689_NUM_, GW7out689+(c+1)*OUTPUT689_NUM_) - (GW7out689+c*OUTPUT689_NUM_);
                auto prob = std::max(  *std::max_element(GW7out+c* OUTPUT_NUM_ , GW7out+(c+1)* OUTPUT_NUM_ ) ,  *std::max_element(GW7out689+c*OUTPUT689_NUM_, GW7out689+(c+1)*OUTPUT689_NUM_)    );
                KeyPoints keypoint;
                x_index/=2;
                y_index/=2;

                if(r_h>r_w){
                    x_index = x_index/r_w;
                    y_index = (y_index - ( INPUT_H_ - r_w * img.rows )/2 )/r_w; 
                }else{

                    x_index = (x_index - ( INPUT_W_ - r_h * img.cols )/2 )/r_h;
                    y_index = y_index/r_h;

                } 
                keypoint.p = cv::Point2f(x_index +obj.rect.x  ,y_index + obj.rect.y);
                keypoint.prob = prob;
                keypoints.push_back(keypoint);
            }


        }else if(label==6 || label==7){
            for(int c = 0 ; c < CLASSESGW11_ ; c++){
                auto x_index = std::max_element(GW11out+c*OUTPUT_NUM_, GW11out+(c+1)*OUTPUT_NUM_) - (GW11out+c*OUTPUT_NUM_);
                auto y_index = std::max_element(GW11out689+c*OUTPUT689_NUM_, GW11out689+(c+1)*OUTPUT689_NUM_) - (GW11out689+c*OUTPUT689_NUM_);
                auto prob = std::max(  *std::max_element(GW11out+c* OUTPUT_NUM_ , GW11out+(c+1)* OUTPUT_NUM_ ) ,  *std::max_element(GW11out689+c*OUTPUT689_NUM_, GW11out689+(c+1)*OUTPUT689_NUM_)    );
                KeyPoints keypoint;
                x_index/=2;
                y_index/=2;

                if(r_h>r_w){
                    x_index = x_index/r_w;
                    y_index = (y_index - ( INPUT_H_ - r_w * img.rows )/2 )/r_w; 
                }else{

                    x_index = (x_index - ( INPUT_W_ - r_h * img.cols )/2 )/r_h;
                    y_index = y_index/r_h;

                }             
                keypoint.p = cv::Point2f(x_index +obj.rect.x  ,y_index + obj.rect.y);
                keypoint.prob = prob;
                keypoints.push_back(keypoint);
            }

        }
        obj.result_kp = keypoints;
   
    }

float Pose::calAngle(cv::Point2f  p1 , cv::Point2f  p2){

        float angle = 0.0;
        float radian_value = acos((p1.x*p2.x+p1.y*p2.y)/(sqrt(p1.x*p1.x+p1.y*p1.y)*sqrt(p2.x*p2.x+p2.y*p2.y)));
        angle = 180*radian_value/3.1415;
        return angle;

}

/**
 * @brief draw rectangle and keypoints
 * GW4: 
 *     0:zuozhidian
 *     1:zuodongdian
 *     2:youzhidian
 *     3:youdongdian
 * skeleton: 
 *     {0,1},{2,3} 
 * GW7: 
 *     0:zuozhidian
 *     1:youzhidian
 *     2:zuodong
 *     3:youdong
 * skeleton: 
 *     {0,1},{2,3} 
 * GW11: 
 *     0:zuo
 *     1:zhong
 *     2:you
 * skeleton: 
 *     {0,1},{1,2} 
**/
void Pose::drawObjects(cv::Mat& image, std::vector<Object>&objects,cv::VideoWriter & writer){

    static const int joint_pairs_GW4_GW7[2][2] = {
        {0, 1}, {2, 3}
    };

    static const int joint_pairs_GW11[2][2] = {
        {0, 1}, {1, 2}
    };


    static const unsigned char colors[81][3] = {
        {56, 0, 255},{226, 255, 0},{0, 94, 255},{0, 37, 255},{0, 255, 94},{255, 226, 0},{0, 18, 255},{255, 151, 0},
        {170, 0, 255},{0, 255, 56},{255, 0, 75},{0, 75, 255},{0, 255, 169},{255, 0, 207},{75, 255, 0},{207, 0, 255},
        {37, 0, 255},{0, 207, 255},{94, 0, 255},{0, 255, 113},{255, 18, 0},{255, 0, 56},{18, 0, 255},{0, 255, 226},
        {170, 255, 0},{255, 0, 245},{151, 255, 0},{132, 255, 0},{75, 0, 255},{151, 0, 255},{0, 151, 255},{132, 0, 255},
        {0, 255, 245},{255, 132, 0},{226, 0, 255},{255, 37, 0},{207, 255, 0},{0, 255, 207},{94, 255, 0},{0, 226, 255},
        {56, 255, 0},{255, 94, 0},{255, 113, 0},{0, 132, 255},{255, 0, 132},{255, 170, 0},{255, 0, 188},{113, 255, 0},
        {245, 0, 255},{113, 0, 255},{255, 188, 0},{0, 113, 255},{255, 0, 0},{0, 56, 255},{255, 0, 113},{0, 255, 188},
        {255, 0, 94},{255, 0, 18},{18, 255, 0},{0, 255, 132},{0, 188, 255},{0, 245, 255},{0, 169, 255},{37, 255, 0},
        {255, 0, 151},{188, 0, 255},{0, 255, 37},{0, 255, 0},{255, 0, 170},{255, 0, 37},{255, 75, 0},{0, 0, 255},
        {255, 207, 0},{255, 0, 226},{255, 245, 0},{188, 255, 0},{0, 255, 18},{0, 255, 75},{0, 255, 151},{255, 56, 0},{245, 255, 0}
    };

    srand((unsigned)time(NULL));
   
    for(int idx = 0 ; idx < objects.size(); idx ++){

        const unsigned char* color = colors[rand() % 81];
        Object obj = objects[idx];
        
        if (obj.prob < 0.15)
            continue;

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

        char text[256];

        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y ;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

        
        std::vector<KeyPoints>keypoints = obj.result_kp;

        if(!keypoints.empty()){

            int label = obj.label;
            float angle;
            bool with_low_score_point = false;

            with_low_score_point = std::any_of(keypoints.begin(),keypoints.end(),[&](auto i){return i.prob<0.3f;});

            if(!with_low_score_point){

                std::string result ;
                if(label==0 || label ==1){
                    
                    angle = calAngle(keypoints[0].p-keypoints[1].p, keypoints[2].p - keypoints[3].p ); 

                    result = std::abs(180 - angle) <= 13 ? "he" :"kai";

                }else if(label== 2 || label ==3){

                    angle = calAngle(keypoints[0].p-keypoints[1].p, keypoints[2].p - keypoints[3].p ); 

                    result = std::abs(angle) <= 13 ? "he" : "kai";

                }else if(label == 6 || label==7){
                    
                    angle = calAngle(keypoints[0].p-keypoints[1].p, keypoints[2].p - keypoints[1].p ); 

                    result = std::abs(angle) >=167 ? "he" : "kai"; 
                }

                int angle_baseLine=0;
                std::string str = "angle = " +std::to_string(int(angle)) + ", result = " + result;
                cv::Size angle_size = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &angle_baseLine);
                int x = obj.rect.x;
                int y = obj.rect.y;
                if (y < 0)
                    y = 0;
                if (x + angle_size.width > image.cols)
                    x = image.cols - angle_size.width;

                cv::rectangle(image, cv::Rect(cv::Point(x, y+2*angle_size.height), cv::Size(angle_size.width, angle_size.height + angle_baseLine)),
                        cv::Scalar(255, 255, 255), -1);

                cv::putText(image,str,cv::Point(x, y+3*angle_size.height),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,132,0));
 
            }

            color = colors[rand() % 81];

            for(int i = 0 ; i < 2 ; i++){
                KeyPoints p1,p2;
                if(keypoints.size()==4){
                    p1 = keypoints[ joint_pairs_GW4_GW7[i][0] ];
                    p2 = keypoints[ joint_pairs_GW4_GW7[i][1] ];
                }else if(keypoints.size()==3){
                    p1 = keypoints[ joint_pairs_GW11[i][0] ];
                    p2 = keypoints[ joint_pairs_GW11[i][1] ];

                }
                if (p1.prob < 0.3f || p2.prob < 0.3f)
                        continue;

                cv::line(image, p1.p, p2.p, cv::Scalar(color[0], color[1], color[2]), 4);

            }

            color = colors[rand() % 81];

            for(int i = 0; i < keypoints.size(); i++){

                KeyPoints keypoint = keypoints[i];

                fprintf(stderr, "%.2f %.2f = %.5f\n", keypoint.p.x, keypoint.p.y, keypoint.prob);

                if (keypoint.prob < 0.3f)
                    continue;

                cv::circle(image, keypoint.p, 3, cv::Scalar(color[0],color[1],color[2]), 4);
            }

        }

    }
            // writer<<image;
        cv::imwrite("./imgout.jpg",image);

}