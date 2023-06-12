from pycocotools.coco import COCO
import numpy as np
import tqdm
import argparse
import os
import shutil
def arg_parser():
    parser = argparse.ArgumentParser('code by rbj')
    parser.add_argument('--annotation_path', type=str,default='/911G/data/cure_images/一楼拷贝数据/up_nei/middle_up_nei/train.json')
    #生成的txt文件保存的目录
    parser.add_argument('--save_labels_path', type=str, default='/911G/data/cure_images/一楼拷贝数据/up_nei/middle_up_nei/yolo_dataset/train/labels')
    args = parser.parse_args(args=[])
    #原网页中是args = parser.parse_args()会报错，改成这个以后解决了
    return args
if __name__ == '__main__':
    root_dir  = os.getcwd()
    train_val_str = ["trian","test"]
    args = arg_parser()
    for t in train_val_str:
        args.annotation_path = os.path.join(root_dir,f"{t}.json")
        args.save_labels_path = os.path.join(root_dir,f"yolo_dataset/{t}/labels")


        if not os.path.exists(args.annotation_path):
            raise Exception('please input right path!')

        if not os.path.exists(args.save_labels_path):
            os.makedirs(args.save_labels_path)

        #create images dir
        images_dir = args.save_labels_path.replace("labels","images")
        os.makedirs(images_dir,exist_ok=True)



        data_source = COCO(annotation_file=args.annotation_path)
        catIds = data_source.getCatIds()
        categories = data_source.loadCats(catIds)
        categories.sort(key=lambda x: x['id'])
        #将label写入my_data_label.names里
        # with open("./data/my_data_label.names", "w") as w:
        #     for index, cat in enumerate(categories):
        #         if index + 1 == len(categories):
        #             w.write(cat.name)
        #         else:
        #             w.write(cat.name + "\n")

        classes = {}
        coco_labels = {}
        coco_labels_inverse = {}
        for c in categories:
            coco_labels[len(classes)] = c['id']
            coco_labels_inverse[c['id']] = len(classes)
            classes[c['name']] = len(classes)

        img_ids = data_source.getImgIds()
        for index, img_id in tqdm.tqdm(enumerate(img_ids), desc='change .json file to .txt file'):
            img_info = data_source.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            height = img_info['height']
            width = img_info['width']

            save_path = os.path.join(args.save_labels_path, file_name.replace("jpg","txt"))

            src_img_path = os.path.join(root_dir,f"{t}/{file_name}")
            dst_img_path = os.path.join(images_dir,f"{file_name}")
            shutil.copy(src_img_path, dst_img_path) 
            
            
            with open(save_path, mode='w') as fp:
                annotation_id = data_source.getAnnIds(img_id)
                # boxes = np.zeros((0, 5))
                if len(annotation_id) == 0:
                    fp.write('')
                    continue
                annotations = data_source.loadAnns(annotation_id)
                lines = ''
                for annotation in annotations:
                    box = annotation['bbox']
                    # some annotations have basically no width / height, skip them
                    if box[2] < 1 or box[3] < 1:
                        continue
                    #top_x,top_y,width,height---->cen_x,cen_y,width,height
                    box[0] = round((box[0] + box[2] / 2) / width, 6)
                    box[1] = round((box[1] + box[3] / 2) / height, 6)
                    box[2] = round(box[2] / width, 6)
                    box[3] = round(box[3] / height, 6)
                    label = coco_labels_inverse[annotation['category_id']]
                    lines = lines + str(label)
                    for i in box:
                        lines += ' ' + str(i)
                    
                    if "keypoints" in annotation:
                        # keypoints = [round(x,3) for x in annotation['keypoints']]
                        keypoints = annotation['keypoints']
                        for i,kp in enumerate(keypoints):
                            if i %3 ==0:
                                lines += ' ' + str(round(kp/width,6))
                            elif i%3 == 1:
                                lines += ' ' + str(round(kp/height,6))
                            else:
                                lines += ' ' + str(kp)


                    lines += '\n'
                fp.writelines(lines)

        

        train_val_dir = os.path.join(os.path.dirname(args.save_labels_path),"train_val_txt")
        os.makedirs(train_val_dir,exist_ok=True)
        with open(os.path.join(train_val_dir,f'{t}.txt'),"w") as f:
            for file_name in os.listdir(args.save_labels_path):
                file_path = os.path.join(args.save_labels_path,file_name)
                js_path = file_path.replace("labels","images").replace("txt","jpg")
                f.write(js_path+"\n")
        

        print('finish')
