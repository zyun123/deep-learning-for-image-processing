#include <iostream>
#include <cstdio>
#include <string>

#include <opencv2/opencv.hpp>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
// only needed for export_method=tracing
#include <torchvision/vision.h> // @oss-only

using namespace std;

class Keypoint_rcnn
{ 
private:
    /* data */
    bool get_module();
    c10::IValue get_tracing_inputs(cv::Mat& img, c10::Device device);
    torch::jit::Module m_module;
    string m_weight_path;
    
public:  
    Keypoint_rcnn(const char* weight_path);
    int Forward(const char* image_file, float* rest);
    int Forward2(int rows, int cols, unsigned char *data, float* rest);
    int m_pred_rows;
    int m_pred_cols; 
};

Keypoint_rcnn::Keypoint_rcnn(const char* weight_path)
{
    this->m_weight_path = weight_path;
    this->m_pred_rows = 0;
    this->m_pred_cols = 0;
    get_module();
}

bool Keypoint_rcnn::get_module()
{
    // auto start_time = chrono::high_resolution_clock::now();
    this->m_module = torch::jit::load(m_weight_path);
    // auto end_time = chrono::high_resolution_clock::now();
    // auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
    //             .count();
    // cout << "Load Weight cost time: " << ms * 1.0 / 1e6 << " seconds" << endl;
    assert(this->m_module.buffers().size() > 0);
    return 0;
}

c10::IValue Keypoint_rcnn::get_tracing_inputs(cv::Mat& img, c10::Device device) {
  const int height = img.rows;
  const int width = img.cols;
  const int channels = 3;

  auto input =
      torch::from_blob(img.data, {height, width, channels}, torch::kUInt8);
  // HWC to CHW
  input = input.to(device, torch::kFloat).permute({2, 0, 1}).contiguous();
  return input;
}

int Keypoint_rcnn::Forward(const char* image_file, float* rest)
{
    torch::jit::getBailoutDepth() = 1;
    torch::autograd::AutoGradMode guard(false);
    auto device = (*begin(this->m_module.buffers())).device();  

    cv::Mat input_img = cv::imread(image_file, cv::IMREAD_COLOR); // Cost 13ms
    // auto start_time = chrono::high_resolution_clock::now();
    auto inputs = get_tracing_inputs(input_img, device);
    // Run the network
    auto output = this->m_module.forward({inputs});
    if (device.is_cuda())
        c10::cuda::getCurrentCUDAStream().synchronize();

    // Parse Keypoint R-CNN outputs
    auto outputs = output.toTuple()->elements();
    torch::Tensor pred_keypoints = outputs[3].toTensor();
    // get  rows and cols of the pred_keypoints
    this->m_pred_rows = pred_keypoints.sizes()[1];
    this->m_pred_cols = pred_keypoints.sizes()[2];
    // get data
    pred_keypoints = pred_keypoints.to(torch::kCPU);
    // auto start_time = chrono::high_resolution_clock::now();
    // cout<<"pred_keypoints"<<pred_keypoints<<endl;
    memcpy(rest, pred_keypoints.data_ptr(), this->m_pred_rows*this->m_pred_cols*sizeof(float));
   
    // auto end_time = chrono::high_resolution_clock::now();
    // auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    // cout << "Copy cost time: " << ms * 1.0 / 1e3 << " ms" << endl;

    return 0;
}

int Keypoint_rcnn::Forward2(int rows, int cols, unsigned char *data, float* rest)
{
    torch::jit::getBailoutDepth() = 1;
    torch::autograd::AutoGradMode guard(false);
    auto device = (*begin(this->m_module.buffers())).device();  

    // auto start_time = chrono::high_resolution_clock::now();
    cv::Mat input_img = cv::Mat(rows, cols, CV_8UC3, data);
    auto inputs = get_tracing_inputs(input_img, device);
    // Run the network
    auto start_time = chrono::high_resolution_clock::now(); 
    auto output = this->m_module.forward({inputs});
    auto end_time = chrono::high_resolution_clock::now();
    if (device.is_cuda())
        c10::cuda::getCurrentCUDAStream().synchronize();

    // Parse Keypoint R-CNN outputs
    auto outputs = output.toTuple()->elements();
    torch::Tensor pred_keypoints = outputs[3].toTensor();
    // get  rows and cols of the pred_keypoints
    this->m_pred_rows = pred_keypoints.sizes()[1];
    this->m_pred_cols = pred_keypoints.sizes()[2];
    // get data
    pred_keypoints = pred_keypoints.to(torch::kCPU);
    // cout<<"pred_keypoints"<<pred_keypoints<<endl;
    memcpy(rest, pred_keypoints.data_ptr(), this->m_pred_rows*this->m_pred_cols*sizeof(float));

    // auto end_time = chrono::high_resolution_clock::now();
    auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    cout << "Copy cost time: " << ms * 1.0 / 1e3 << " ms" << endl;
    return 0;
}


// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
    Keypoint_rcnn* keypoints(const char* weight_path) {
        return new Keypoint_rcnn(weight_path);
    }

    int forward(Keypoint_rcnn* kpr, const char* image_file, float* rest)
    {   
        return kpr->Forward(image_file, rest);
    }

    int forward2(Keypoint_rcnn* kpr, int rows, int cols, unsigned char *data, float* rest)
    {   
        return kpr->Forward2(rows, cols, data, rest);
    }

    int getCols(Keypoint_rcnn* kpr)
    {
        return  kpr->m_pred_cols;
    }

    int getRows(Keypoint_rcnn* kpr)
    {
        return kpr->m_pred_rows;
    }

    // 释放对象内存函数
    void del_Keypoint_rcnn(Keypoint_rcnn* kpr) {
        delete kpr;
    }

    
}
