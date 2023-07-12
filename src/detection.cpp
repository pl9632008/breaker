#include "detection.h"

static float yoloin[3*640*640];
static float yoloout[25200*13];


void Detection::loadEngine(std::string path){

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

        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger_));
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(trtModelStream,size));
        std::unique_ptr<IExecutionContext>context(engine->createExecutionContext());

        runtime_ = std::move(runtime);
        engine_  = std::move(engine);
        context_ = std::move(context);

        delete[] trtModelStream;
}

void Detection::doInference(cv::Mat & org_img){

        int32_t images_index = engine_->getBindingIndex(images_);
        int32_t output0_index = engine_->getBindingIndex(output0_);
        void * buffers[2];
        cudaMalloc(&buffers[images_index],BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float));
        cudaMalloc(&buffers[output0_index],BATCH_SIZE_*NUM_*CLASSES_*sizeof(float));

        cv::Mat pr_img = preprocessImg(org_img,INPUT_W_,INPUT_H_);
        
        for(int i = 0 ; i < INPUT_W_*INPUT_H_;i++){
            yoloin[i] = pr_img.at<cv::Vec3b>(i)[2]/255.0;
            yoloin[i+INPUT_W_*INPUT_H_] = pr_img.at<cv::Vec3b>(i)[1]/255.0;
            yoloin[i+2*INPUT_W_*INPUT_H_]=pr_img.at<cv::Vec3b>(i)[0]/255.0;
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaMemcpyAsync(buffers[images_index],yoloin,BATCH_SIZE_*CHANNELS_*INPUT_W_*INPUT_H_*sizeof(float),cudaMemcpyHostToDevice,stream);
        context_->enqueueV2(buffers,stream, nullptr);
        cudaMemcpyAsync(yoloout,buffers[output0_index],BATCH_SIZE_*NUM_*CLASSES_*sizeof(float),cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(buffers[images_index]);
        cudaFree(buffers[output0_index]);


}

cv::Mat Detection::preprocessImg(cv::Mat& img, int input_w, int input_h){
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

void Detection::qsort_descent_inplace(std::vector<Object>&faceobjects,int left, int right){
    int i = left;
    int j = right;
    float p = faceobjects[(left+right)/2].prob;
    while (i<=j){
        while (faceobjects[i].prob>p ){
            i++;
        }
        while (faceobjects[j].prob<p){
            j--;
        }
        if(i<=j){
            std::swap(faceobjects[i],faceobjects[j]);
            i++;
            j--;
        }

    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

void  Detection::qsort_descent_inplace(std::vector<Object>&faceobjects){
    if(faceobjects.empty()){
        return ;
    }
    qsort_descent_inplace(faceobjects,0,faceobjects.size()-1);
}

float Detection::intersection_area(Object & a,Object&b) {
    cv::Rect2f inter = a.rect&b.rect;
    return inter.area();

}

void Detection::nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
         Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
          Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}




std::vector<Object> Detection::decode(){

    std::vector<Object> objects;
    for(int i = 0 ; i<NUM_;i++){
        if(yoloout[CLASSES_*i+4]>0.3){
            int l,r,t,b;
            float r_w = INPUT_W_/(org_img_cols_*1.0);
            float r_h = INPUT_H_/(org_img_rows_*1.0);

            float x = yoloout[CLASSES_*i+0];
            float y = yoloout[CLASSES_*i+1];
            float w = yoloout[CLASSES_*i+2];
            float h = yoloout[CLASSES_*i+3];
            float score = yoloout[CLASSES_*i+4];

            if(r_h>r_w){
                l = x-w/2.0;
                r = x+w/2.0;
                t = y-h/2.0-(INPUT_H_-r_w*org_img_rows_)/2;
                b = y+h/2.0-(INPUT_H_-r_w*org_img_rows_)/2;
                l=l/r_w;
                r=r/r_w;
                t=t/r_w;
                b=b/r_w;
            }else{
                l = x-w/2.0-(INPUT_W_-r_h*org_img_cols_)/2;
                r = x+w/2.0-(INPUT_W_-r_h*org_img_cols_)/2;
                t = y-h/2.0;
                b = y+h/2.0;
                l=l/r_h;
                r=r/r_h;
                t=t/r_h;
                b=b/r_h;
            }
            int label_index = std::max_element(yoloout+CLASSES_*i+5,yoloout+CLASSES_*(i+1)) - (yoloout+CLASSES_*i+5);
         
            Object obj;
            obj.rect.x = std::max(l,0);
            obj.rect.y = std::max(t,0);
            obj.rect.width=r-l;
            obj.rect.height=b-t;
            if(obj.rect.x+obj.rect.width>org_img_cols_){
                obj.rect.width = org_img_cols_ - obj.rect.x;
            }
            if(obj.rect.y+obj.rect.height>org_img_rows_){
                obj.rect.height = org_img_rows_ - obj.rect.y;
            }
            obj.label = label_index;
            obj.prob = score;
            objects.push_back(obj);
            
        }

    }
    qsort_descent_inplace(objects);
    std::vector<int> picked;
    nms_sorted_bboxes(objects,picked,0.45);
    int count = picked.size();
    std::cout<<"count="<<count<<std::endl;
    std::vector<Object>obj_out(count);
    for(int i = 0 ; i <count ; ++i){
        obj_out[i] = objects[picked[i]];
    }
    return obj_out;
}