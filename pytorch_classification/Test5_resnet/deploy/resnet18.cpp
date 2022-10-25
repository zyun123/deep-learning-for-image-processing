#include <torch/script.h>
#include<iostream>
// #include<fstream>
// #include<string>
// #include<memory>
// #include<math.h>
#include<opencv2/opencv.hpp>

void readTxt(std::string &file,std::vector<std::string> &clas){
    std::ifstream infile;
    infile.open(file,std::ios::in);
    assert(infile.is_open());
    std::string s;
    while(getline(infile,s)){
        std::cout << s <<std::endl;
        clas.push_back(s);    
    }
    infile.close();
}


std::vector<float> softmax(std::vector<float> &input){
    float total = 0;
    float MAX = 0;
    for(auto x:input){
        MAX = std::max(x,MAX);
    }
    for (auto x:input){
        total+=std::exp(x-MAX);
    }
    std::vector<float> result;
    for(auto x:input){
        result.push_back(exp(x-MAX)/total);
    }
    
    return result;
}



int main(int argc,char** argv){
    if (argc !=2){
        std::cerr << "usage: resnet18 <path_to_exported_script_module>\n";
        return -1;
    }
    // torch::jit::Module module;
    torch::jit::script::Module module;
    try{
        module = torch::jit::load(argv[1]);
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "ok\n";
    std::string path = "../class_indices.txt";
    std::vector<std::string> clas;
    readTxt(path,clas);
    std::cout << "vector clas: " << clas <<std::endl;
    //create a vector of inputs 
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(torch::ones({1,3,224,224}));
    //读取文件下所有的图片
    std::string img_dir = "/home/zy/vision/deep-learning-for-image-processing/pytorch_classification/Test5_resnet/images";
    std::vector<cv::String> img_lists;
    cv::glob(img_dir,img_lists,true);
    // torch::jit::getBailoutDepth() = 1;
    // torch::autograd::AutoGradMode guard(false);
    for(auto name:img_lists){
        std::cout << name << std::endl;
        cv::Mat src = cv::imread(name,cv::IMREAD_COLOR);
        const int channels = src.channels(),height = src.rows,width = src.cols;
        cv::resize(src,src,cv::Size(224,224));
        // src.convertTo(src,CV_32F);
        // src = src/255.0;
        // cv::subtract(src,cv::Scalar(0.485, 0.456, 0.406),src);
        // cv::divide(src,cv::Scalar(0.229, 0.224, 0.225),src);
        
        /*
        float *data = (float*)malloc(channels*height*width*sizeof(float));
        memset(data,0,channels*height*width*sizeof(float));
        int image_size = height * width;
        for (size_t row = 0; row < height; row++) {
		for (size_t col = 0; col < width; col++) {
			for (size_t ch = 0; ch < channels; ch++) {
				data[image_size*ch + row*width + col] = src.at<cv::Vec3f>(row, col)[ch];
			}
		}
        }
        */
        // std::vector<torch::jit::IValue> inputs;
        auto input = torch::from_blob(src.data,{1,height,width,channels},torch::kUInt8);

        // input = input.permute({0,3,1,2}).contiguous();
        input = input.to(torch::kFloat).permute({0,3,1,2}).contiguous();
        input = input.div(255);
    
        auto input1 = torch::ones({1,3,224,224});

        // inputs.push_back(input);
        auto output = module.forward({input1}).toTensor();
        std::cout << "output: " <<output << std::endl;

        std::vector<float> output2(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
        std::vector<float> softmax_out;
        softmax_out = softmax(output2);
        float final_max = 0;
        int max_index = 0;
        for (int i = 0;i<softmax_out.size();i++){
            if(softmax_out[i] >final_max){
                final_max = softmax_out[i];
                max_index = i;
            }     
        }
        
        std::cout <<"softmax_out: " << softmax_out<<std::endl;
        std::cout << "max_index: " << max_index<<std::endl;
        std::cout << "predict image perosn pose:" << clas[max_index] << std::endl;
        cv::putText(src,clas[max_index].c_str(),cv::Point(50,50),cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
        cv::imshow("src",src);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    // cv::destroyAllWindows();



    // // inputs.to(at::kCUDA);
    // //execute the model and turn its output into a tensor
    // at::Tensor output = module.forward(inputs).toTensor();
    // //将at::Tensor 转成vector
    // std::vector<float> output2(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
    // //softmax
    // std::vector<float> softmax_out;
    // softmax_out = softmax(output2);
    // // auto predict = softmax(output);
    // // auto predict_cla = torch::argmax(predict);
    // // std::cout << "predict_cla:  " << predict_cla << std::endl;

    // float final_max = 0;
    // int max_index = 0;
    // for (int i = 0;i<softmax_out.size();i++){
    //     if(softmax_out[i] >final_max){
    //         final_max = softmax_out[i];
    //         max_index = i;
    //     }     
    // }
    // // std::cout<< output.slice(/*dim=*/0,/*start=*/0,/*end=*/10);
    // std::cout << sizeof(output.data_ptr())/sizeof(float)<<std::endl;
    // std::cout << softmax_out<<std::endl;
    // std::cout << final_max<<std::endl;
    // std::cout << max_index<<std::endl;
}