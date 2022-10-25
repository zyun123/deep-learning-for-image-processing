#include <torch/script.h>
#include<iostream>
#include<memory>
#include<math.h>
std::vector<float> softmax(std::vector<float> input){
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
    torch::jit::script::Module module;
    try{
        module = torch::jit::load(argv[1]);
    }
    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "ok\n";


    //create a vector of inputs 
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1,3,224,224}));
    // inputs.to(at::kCUDA);
    //execute the model and turn its output into a tensor
    at::Tensor output = module.forward(inputs).toTensor();
    //将at::Tensor 转成vector
    std::vector<float> output2(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
    //softmax
    std::vector<float> softmax_out;
    softmax_out = softmax(output2);
    // auto predict = softmax(output);
    // auto predict_cla = torch::argmax(predict);
    // std::cout << "predict_cla:  " << predict_cla << std::endl;

    float final_max = 0;
    int max_index = 0;
    for (int i = 0;i<softmax_out.size();i++){
        if(softmax_out[i] >final_max){
            final_max = softmax_out[i];
            max_index = i;
        }     
    }
    // std::cout<< output.slice(/*dim=*/0,/*start=*/0,/*end=*/10);
    std::cout << output<<std::endl;
    std::cout << softmax_out<<std::endl;
    std::cout << final_max<<std::endl;
    std::cout << max_index<<std::endl;

}