import os
import json
kpnames = ["L-pi-1", "L-pi-2", "L-pi-3", "L-pi-4", "L-pi-5", "L-pi-6", "L-pi-7", "L-pi-8", "L-pi-9", "L-pi-10", "L-pi-11", "L-pi-12", 
        "R-pi-1", "R-pi-2", "R-pi-3", "R-pi-4", "R-pi-5", "R-pi-6", "R-pi-7", "R-pi-8", "R-pi-9", "R-pi-10", "R-pi-11", "R-pi-12", 
        "R-xinbao-1", "R-xinbao-2", "R-xinbao-3", "R-xinbao-4", "R-xinbao-5", "R-xinbao-6", "R-xinbao-7", "R-xinbao-8", "R-xinbao-9", 
        "L-xinbao-1", "L-xinbao-2", "L-xinbao-3", "L-xinbao-4", "L-xinbao-5", "L-xinbao-6", "L-xinbao-7", "L-xinbao-8", "L-xinbao-9", 
        "L-wei-15", "L-wei-16", "L-wei-17", "L-wei-18", "L-wei-19", "L-wei-20", "L-wei-21", "L-wei-22", "L-wei-23", "L-wei-24", "L-wei-25", "L-wei-26", "L-wei-27", "L-wei-28", "L-wei-29", "L-wei-30", 
        "R-wei-15", "R-wei-16", "R-wei-17", "R-wei-18", "R-wei-19", "R-wei-20", "R-wei-21", "R-wei-22", "R-wei-23", "R-wei-24", "R-wei-25", "R-wei-26", "R-wei-27", "R-wei-28", "R-wei-29", "R-wei-30", 
        "L-fei-1", "L-fei-2", "L-fei-3", "L-fei-4", "L-fei-5", "L-fei-6", "L-fei-7", "L-fei-8", 
        "R-fei-1", "R-fei-2", "R-fei-3", "R-fei-4", "R-fei-5", "R-fei-6", "R-fei-7", "R-fei-8"]

def cal_uper_lower(jsfile):
    with open(jsfile,"r") as f:
        data = json.load(f)

    shapes = data["shapes"]
    
    uper = []
    lower = []
    for i,kp in enumerate(kpnames):
        for shape in shapes:
            if shape['label'] == kp:
                py= shape['points'][0][1]
                if py > 470:
                    lower.append(i)
                else:
                    uper.append(i)

    print("uper:",uper)
    print("lower:",lower)


def cal_flip_pairs(jfsiles):
    pairs = []
    basenames = ["pi-1", "pi-2", "pi-3", "pi-4", "pi-5", "pi-6", "pi-7", "pi-8", "pi-9", "pi-10", "pi-11", "pi-12",
    "xinbao-1", "xinbao-2", "xinbao-3", "xinbao-4", "xinbao-5", "xinbao-6", "xinbao-7", "xinbao-8", "xinbao-9", 
    "wei-15", "wei-16", "wei-17", "wei-18", "wei-19", "wei-20", "wei-21", "wei-22", "wei-23", "wei-24", "wei-25", "wei-26", "wei-27", "wei-28", "wei-29", "wei-30", 
    "fei-1", "fei-2", "fei-3", "fei-4", "fei-5", "fei-6", "fei-7", "fei-8"]
    for kpname in basenames:
        lkpname = "L-" + kpname
        rkpname = "R-" + kpname
        pairs.append([kpnames.index(lkpname),kpnames.index(rkpname)])

    print(pairs)
    print("len(pairs) = " + str(len(pairs)))
        






if __name__ == '__main__':

    jsfile = "/911G/data/temp/20221229新加手托脚托新数据/精确标注494套middle_up_nei_changerec/hrnet_data_rotate90/m_up_nei_20221228151246667.json"
    # cal_uper_lower(jsfile)
    cal_flip_pairs(jsfile)



