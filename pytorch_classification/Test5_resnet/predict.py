import glob
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize(224),
        #  transforms.CenterCrop(224),
         transforms.ToTensor(),
        #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    # #use one image to predict
    # # load image
    # # img_path = "/911G/data/person_cls/person_data/val/up/000140.jpg"
    # img_path = "images/2022_3_16_16_53_16_camera0.jpg"
    # # img_path = "images/camera0120.jpg"
    # # img_path = "images/camera0143.jpg"
    # # img_path = "images/camera0155.jpg"
    # # img_path = "images/camera0159.jpg"    #预测不准确
    # # img_path = "images/camera0167.jpg"
    # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path)
    # plt.imshow(img)
    # # [N, C, H, W]
    # img = data_transform(img)
    # # expand batch dimension
    # img = torch.unsqueeze(img, dim=0)

    # # read class_indict
    # json_path = './class_indices.json'
    # assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # with open(json_path, "r") as f:
    #     class_indict = json.load(f)

    # # create model
    # model = resnet34(num_classes=2).to(device)

    # # load model weights
    # weights_path = "./resNet34.pth"
    # assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # model.load_state_dict(torch.load(weights_path, map_location=device))

    # # prediction
    # model.eval()
    # with torch.no_grad():
    #     # predict class
    #     output = torch.squeeze(model(img.to(device))).cpu()
    #     print("output:{}".format(output))
    #     predict = torch.softmax(output, dim=0)
    #     predict_cla = torch.argmax(predict).numpy()
    # print("cla index:  {}".format(predict_cla))

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    # plt.title(print_res)
    # for i in range(len(predict)):
    #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
    #                                               predict[i].numpy()))
    # # plt.show()
    # plt.savefig("output/{}".format(os.path.basename(img_path)))
    # print("predict done")



    #use multi images to predict
    img_list=  glob.glob("/home/zy/Downloads/middle_down_wai_20221207161938168等2个文件/*.jpg")
    for img_path in img_list:
        print("-"*30)
        print(img_path)
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = resnet34(num_classes=2).to(device)

        # load model weights
        weights_path = "./resNet34.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            print("output:{}".format(output))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        # print("cla index:  {}".format(predict_cla))
        print("softmax output",predict)

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                    predict[predict_cla].numpy())
        plt.title(print_res)

        # for i in range(len(predict)):
        #     print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
        #                                             predict[i].numpy()))
        # plt.show()
        f = plt.gcf()
        f.savefig("output/{}".format(os.path.basename(img_path)))
        f.clear()
        print("predict done")






if __name__ == '__main__':
    main()
