


__all__ = ["COCO_PERSON_KEYPOINT_NAMES_DOWN","KEYPOINT_CONNECTION_RULES_DOWN",
            "COCO_PERSON_KEYPOINT_NAMES_UP","KEYPOINT_CONNECTION_RULES_UP",
            "COCO_PERSON_KEYPOINT_NAMES_LEFT_DOWN","KEYPOINT_CONNECTION_RULES_LEFT_DOWN",
            "COCO_PERSON_KEYPOINT_NAMES_HEAD_MIDDLE_DOWN","KEYPOINT_CONNECTION_RULES_HEAD_MIDDLE_DOWN",
            "KEYPOINT_CONNECTION_RULES_WHOLE_DOWN"]


#全经络版本84个点            
# COCO_PERSON_KEYPOINT_NAMES_UP = ['R-shen-14', 'R-shen-15', 'R-shen-16', 'R-shen-17', 'R-shen-18', 'R-shen-19',
#                          'L-shen-14', 'L-shen-15', 'L-shen-16', 'L-shen-17', 'L-shen-18', 'L-shen-19','R-xinbao-1', 
#                          'R-xinbao-2', 'R-xinbao-3', 'R-xinbao-4', 'R-xinbao-5', 'R-xinbao-6','R-xinbao-7', 
#                          'R-xinbao-8', 'R-xinbao-9','L-xinbao-1', 'L-xinbao-2', 'L-xinbao-3', 'L-xinbao-4', 
#                          'L-xinbao-5', 'L-xinbao-6','L-xinbao-7', 'L-xinbao-8', 'L-xinbao-9','L-fei-1', 'L-fei-2', 
#                          'L-fei-3', 'L-fei-4', 'L-fei-5', 'L-fei-6', 'L-fei-7', 'L-fei-8','R-fei-1', 'R-fei-2',
#                          'R-fei-3', 'R-fei-4', 'R-fei-5', 'R-fei-6', 'R-fei-7', 'R-fei-8','L-wei-15', 'L-wei-16', 
#                          'L-wei-17', 'L-wei-18', 'L-wei-19', 'L-wei-20', 'L-wei-21', 'L-wei-22', 'L-wei-23',
#                             'L-wei-24', 'L-wei-25', 'L-wei-26', 'L-wei-27', 'L-wei-28', 'L-wei-29', 'L-wei-30', 
#                             'R-wei-15', 'R-wei-16', 'R-wei-17', 'R-wei-18', 'R-wei-19', 'R-wei-20', 'R-wei-21', 
#                             'R-wei-22', 'R-wei-23','R-wei-24', 'R-wei-25', 'R-wei-26', 'R-wei-27', 'R-wei-28', 
#                             'R-wei-29', 'R-wei-30','L-dan-26','L-dan-51','L-dan-52','R-dan-26','R-dan-51','R-dan-52']


#全经络版本  90个点                          
# COCO_PERSON_KEYPOINT_NAMES_DOWN = ['R-dachang-1','R-dachang-2', 'R-dachang-3','R-dachang-4','R-dachang-5',
# 'R-dachang-6','R-dachang-7','R-dachang-8',
#     'L-dachang-1','L-dachang-2','L-dachang-3','L-dachang-4','L-dachang-5','L-dachang-6',
#     'L-dachang-7','L-dachang-8','L-sanjiao-1','L-sanjiao-2', 'L-sanjiao-3','L-sanjiao-4','L-sanjiao-5',
#     'L-sanjiao-6','L-sanjiao-7','R-sanjiao-1','R-sanjiao-2', 'R-sanjiao-3','R-sanjiao-4','R-sanjiao-5',
#     'R-sanjiao-6','R-sanjiao-7','L-xiaochang-1', 'L-xiaochang-2','L-xiaochang-3','L-xiaochang-4','L-xiaochang-5',
#     'L-xiaochang-6','L-xiaochang-7','L-xiaochang-8','L-xiaochang-9','L-xiaochang-10',
#     'L-xiaochang-11', 'L-xiaochang-12',
#     'R-xiaochang-1', 'R-xiaochang-2','R-xiaochang-3','R-xiaochang-4','R-xiaochang-5',
#     'R-xiaochang-6','R-xiaochang-7','R-xiaochang-8','R-xiaochang-9','R-xiaochang-10',
#     'R-xiaochang-11', 'R-xiaochang-12','L-pangguang-9','L-pangguang-10', 'L-pangguang-11','L-pangguang-12','L-pangguang-13',
#     'L-pangguang-14','L-pangguang-15', 'L-pangguang-16','L-pangguang-17','L-pangguang-18',
#     'L-pangguang-19','L-pangguang-20', 'L-pangguang-21','L-pangguang-22','L-pangguang-23',
#     'L-pangguang-24','L-pangguang-25','L-pangguang-26','R-pangguang-9','R-pangguang-10', 
#     'R-pangguang-11','R-pangguang-12','R-pangguang-13',
#     'R-pangguang-14','R-pangguang-15', 'R-pangguang-16','R-pangguang-17','R-pangguang-18',
#     'R-pangguang-19','R-pangguang-20', 'R-pangguang-21','R-pangguang-22','R-pangguang-23',
#     'R-pangguang-24','R-pangguang-25','R-pangguang-26']    
#                     
COCO_PERSON_KEYPOINT_NAMES_DOWN = ['L-sanjiao-1', 'L-sanjiao-2', 'L-sanjiao-3', 'L-sanjiao-4', 'L-sanjiao-5',
                'L-sanjiao-6', 'L-sanjiao-7', 'L-sanjiao-8', 'L-sanjiao-9', 'R-sanjiao-1', 'R-sanjiao-2',
                 'R-sanjiao-3', 'R-sanjiao-4', 'R-sanjiao-5','R-sanjiao-6', 'R-sanjiao-7', 'R-sanjiao-8', 'R-sanjiao-9',
                'L-pangguang-9', 'L-pangguang-10', 'L-pangguang-11', 'L-pangguang-12', 'L-pangguang-13',
                'L-pangguang-14', 'L-pangguang-15', 'L-pangguang-16', 'L-pangguang-17', 'L-pangguang-18',
                'L-pangguang-19', 'L-pangguang-20', 'L-pangguang-21', 'L-pangguang-22', 'L-pangguang-23',
                'L-pangguang-24', 'R-pangguang-9', 'R-pangguang-10',
                'R-pangguang-11', 'R-pangguang-12', 'R-pangguang-13',
                'R-pangguang-14', 'R-pangguang-15', 'R-pangguang-16', 'R-pangguang-17', 'R-pangguang-18',
                'R-pangguang-19', 'R-pangguang-20', 'R-pangguang-21', 'R-pangguang-22', 'R-pangguang-23',
                'R-pangguang-24']


# #加肺经
COCO_PERSON_KEYPOINT_NAMES_UP = ['L-pi-1', 'L-pi-2', 'L-pi-3', 'L-pi-4', 'L-pi-5', 'L-pi-6', 'L-pi-7', 'L-pi-8', 'L-pi-9', 
'L-pi-10', 'L-pi-11', 'L-pi-12', 'R-pi-1', 'R-pi-2', 'R-pi-3', 'R-pi-4', 'R-pi-5', 'R-pi-6', 'R-pi-7', 'R-pi-8', 'R-pi-9', 
'R-pi-10', 'R-pi-11', 'R-pi-12', 'R-xinbao-1', 'R-xinbao-2', 'R-xinbao-3', 'R-xinbao-4', 'R-xinbao-5', 'R-xinbao-6', 
'R-xinbao-7', 'R-xinbao-8', 'R-xinbao-9', 'L-xinbao-1', 'L-xinbao-2', 'L-xinbao-3', 'L-xinbao-4', 'L-xinbao-5', 
'L-xinbao-6', 'L-xinbao-7', 'L-xinbao-8', 'L-xinbao-9', 'L-wei-15', 'L-wei-16', 'L-wei-17', 'L-wei-18', 'L-wei-19', 
'L-wei-20', 'L-wei-21', 'L-wei-22', 'L-wei-23', 'L-wei-24', 'L-wei-25', 'L-wei-26', 'L-wei-27', 'L-wei-28', 'L-wei-29', 
'L-wei-30', 'R-wei-15', 'R-wei-16', 'R-wei-17', 'R-wei-18', 'R-wei-19', 'R-wei-20', 'R-wei-21', 'R-wei-22', 'R-wei-23', 
'R-wei-24', 'R-wei-25', 'R-wei-26', 'R-wei-27', 'R-wei-28', 'R-wei-29', 'R-wei-30','L-fei-1', 'L-fei-2', 'L-fei-3', 
'L-fei-4', 'L-fei-5', 'L-fei-6', 'L-fei-7', 'L-fei-8','R-fei-1', 'R-fei-2', 'R-fei-3', 'R-fei-4', 'R-fei-5', 'R-fei-6', 
'R-fei-7', 'R-fei-8']

#不加肺经
# COCO_PERSON_KEYPOINT_NAMES_UP = ['L-pi-1', 'L-pi-2', 'L-pi-3', 'L-pi-4', 'L-pi-5', 'L-pi-6', 'L-pi-7', 'L-pi-8', 'L-pi-9', 'L-pi-10', 'L-pi-11', 'L-pi-12', 'R-pi-1', 'R-pi-2', 'R-pi-3', 'R-pi-4', 'R-pi-5', 'R-pi-6', 'R-pi-7', 'R-pi-8', 'R-pi-9', 'R-pi-10', 'R-pi-11', 'R-pi-12', 'R-xinbao-1', 'R-xinbao-2', 'R-xinbao-3', 'R-xinbao-4', 'R-xinbao-5', 'R-xinbao-6', 'R-xinbao-7', 'R-xinbao-8', 'R-xinbao-9', 'L-xinbao-1', 'L-xinbao-2', 'L-xinbao-3', 'L-xinbao-4', 'L-xinbao-5', 'L-xinbao-6', 'L-xinbao-7', 'L-xinbao-8', 'L-xinbao-9', 'L-wei-15', 'L-wei-16', 'L-wei-17', 'L-wei-18', 'L-wei-19', 'L-wei-20', 'L-wei-21', 'L-wei-22', 'L-wei-23', 'L-wei-24', 'L-wei-25', 'L-wei-26', 'L-wei-27', 'L-wei-28', 'L-wei-29', 'L-wei-30', 'R-wei-15', 'R-wei-16', 'R-wei-17', 'R-wei-18', 'R-wei-19', 'R-wei-20', 'R-wei-21', 'R-wei-22', 'R-wei-23', 'R-wei-24', 'R-wei-25', 'R-wei-26', 'R-wei-27', 'R-wei-28', 'R-wei-29', 'R-wei-30']





KEYPOINT_CONNECTION_RULES_DOWN = [
        # sanjiao
        ('R-sanjiao-1', 'R-sanjiao-2', (255, 0 , 255)),
        ('R-sanjiao-2', 'R-sanjiao-3', (255, 0 , 255)),
        ('R-sanjiao-3', 'R-sanjiao-4', (255, 0 , 255)),
        ('R-sanjiao-4', 'R-sanjiao-5', (255, 0 , 255)),
        ('R-sanjiao-5', 'R-sanjiao-6', (255, 0 , 255)),
        ('R-sanjiao-6', 'R-sanjiao-7', (255, 0 , 255)),
        ('R-sanjiao-7', 'R-sanjiao-8', (255, 0 , 255)),
        ('R-sanjiao-8', 'R-sanjiao-9', (255, 0 , 255)),

        ('L-sanjiao-1', 'L-sanjiao-2', (255, 0 , 255)),
        ('L-sanjiao-2', 'L-sanjiao-3', (255, 0 , 255)),
        ('L-sanjiao-3', 'L-sanjiao-4', (255, 0 , 255)),
        ('L-sanjiao-4', 'L-sanjiao-5', (255, 0 , 255)),
        ('L-sanjiao-5', 'L-sanjiao-6', (255, 0 , 255)),
        ('L-sanjiao-6', 'L-sanjiao-7', (255, 0 , 255)),
        ('L-sanjiao-7', 'L-sanjiao-8', (255, 0 , 255)),
        ('L-sanjiao-8', 'L-sanjiao-9', (255, 0 , 255)),
        #pangguang
        ('L-pangguang-9', 'L-pangguang-10',  (0,255,255)),
        ('L-pangguang-10', 'L-pangguang-11', (0,255,255)),
        ('L-pangguang-11', 'L-pangguang-12', (0,255,255)),
        ('L-pangguang-12', 'L-pangguang-13', (0,255,255)),
        ('L-pangguang-13', 'L-pangguang-14', (0,255,255)),
        ('L-pangguang-14', 'L-pangguang-15', (0,255,255)),
        ('L-pangguang-15', 'L-pangguang-16', (0,255,255)),
        ('L-pangguang-16', 'L-pangguang-17', (0,255,255)),

        ('L-pangguang-18', 'L-pangguang-19', (255,255,0)),
        ('L-pangguang-19', 'L-pangguang-20', (255,255,0)),
        ('L-pangguang-20', 'L-pangguang-21', (255,255,0)),
        ('L-pangguang-21', 'L-pangguang-22', (255,255,0)),
        ('L-pangguang-22', 'L-pangguang-23', (255,255,0)),
        ('L-pangguang-23', 'L-pangguang-24', (255,255,0)),

        ('R-pangguang-9', 'R-pangguang-10',  (0,255,255)),
        ('R-pangguang-10', 'R-pangguang-11', (0,255,255)),
        ('R-pangguang-11', 'R-pangguang-12', (0,255,255)),
        ('R-pangguang-12', 'R-pangguang-13', (0,255,255)),
        ('R-pangguang-13', 'R-pangguang-14', (0,255,255)),
        ('R-pangguang-14', 'R-pangguang-15', (0,255,255)),
        ('R-pangguang-15', 'R-pangguang-16', (0,255,255)),
        ('R-pangguang-16', 'R-pangguang-17', (0,255,255)),

        ('R-pangguang-18', 'R-pangguang-19', (255,255,0)),
        ('R-pangguang-19', 'R-pangguang-20', (255,255,0)),
        ('R-pangguang-20', 'R-pangguang-21', (255,255,0)),
        ('R-pangguang-21', 'R-pangguang-22', (255,255,0)),
        ('R-pangguang-22', 'R-pangguang-23', (255,255,0)),
        ('R-pangguang-23', 'R-pangguang-24', (255,255,0)),    
    ]
KEYPOINT_CONNECTION_RULES_UP = [
        # xinbao
        ('R-xinbao-1', 'R-xinbao-2', (10, 204, 255)),
        ('R-xinbao-2', 'R-xinbao-3', (20, 204, 255)),
        ('R-xinbao-3', 'R-xinbao-4', (30, 204, 255)),
        ('R-xinbao-4', 'R-xinbao-5', (40, 204, 255)),
        ('R-xinbao-5', 'R-xinbao-6', (50, 204, 255)),
        ('R-xinbao-6', 'R-xinbao-7', (60, 204, 255)),
        ('R-xinbao-7', 'R-xinbao-8', (70, 204, 255)),
        ('R-xinbao-8', 'R-xinbao-9', (80, 204, 255)),

        ('L-xinbao-1', 'L-xinbao-2', (10, 204, 255)),
        ('L-xinbao-2', 'L-xinbao-3', (20, 204, 255)),
        ('L-xinbao-3', 'L-xinbao-4', (30, 204, 255)),
        ('L-xinbao-4', 'L-xinbao-5', (40, 204, 255)),
        ('L-xinbao-5', 'L-xinbao-6', (50, 204, 255)),
        ('L-xinbao-6', 'L-xinbao-7', (60, 204, 255)),
        ('L-xinbao-7', 'L-xinbao-8', (70, 204, 255)),
        ('L-xinbao-8', 'L-xinbao-9', (80, 204, 255)),

        #wei
        ('L-wei-15', 'L-wei-16', (255, 128, 0)),
        ('L-wei-16', 'L-wei-17', (255, 128, 0)),
        ('L-wei-17', 'L-wei-18', (255, 128, 0)),
        ('L-wei-18', 'L-wei-19', (255, 128, 0)),
        ('L-wei-19', 'L-wei-20', (255, 128, 0)),
        ('L-wei-20', 'L-wei-21', (255, 128, 0)),
        ('L-wei-21', 'L-wei-22', (255, 128, 0)),
        ('L-wei-22', 'L-wei-23', (255, 128, 0)),
        ('L-wei-23', 'L-wei-24', (255, 128, 0)),
        ('L-wei-24', 'L-wei-25', (255, 128, 0)),
        ('L-wei-25', 'L-wei-26', (255, 128, 0)),
        ('L-wei-26', 'L-wei-27', (255, 128, 0)),
        ('L-wei-27', 'L-wei-28', (255, 128, 0)),
        ('L-wei-28', 'L-wei-29', (255, 128, 0)),
        ('L-wei-29', 'L-wei-30', (255, 128, 0)),

        ('R-wei-15', 'R-wei-16', (255, 128, 0)),
        ('R-wei-16', 'R-wei-17', (255, 128, 0)),
        ('R-wei-17', 'R-wei-18', (255, 128, 0)),
        ('R-wei-18', 'R-wei-19', (255, 128, 0)),
        ('R-wei-19', 'R-wei-20', (255, 128, 0)),
        ('R-wei-20', 'R-wei-21', (255, 128, 0)),
        ('R-wei-21', 'R-wei-22', (255, 128, 0)),
        ('R-wei-22', 'R-wei-23', (255, 128, 0)),
        ('R-wei-23', 'R-wei-24', (255, 128, 0)),
        ('R-wei-24', 'R-wei-25', (255, 128, 0)),
        ('R-wei-25', 'R-wei-26', (255, 128, 0)),
        ('R-wei-26', 'R-wei-27', (255, 128, 0)),
        ('R-wei-27', 'R-wei-28', (255, 128, 0)),
        ('R-wei-28', 'R-wei-29', (255, 128, 0)),
        ('R-wei-29', 'R-wei-30', (255, 128, 0)),

        #pi
        ('L-pi-1', 'L-pi-2', (10,255,10)),
        ('L-pi-2', 'L-pi-3', (10,255,10)),
        ('L-pi-3', 'L-pi-4', (10,255,10)),
        ('L-pi-4', 'L-pi-5', (10,255,10)),
        ('L-pi-5', 'L-pi-6', (10,255,10)),
        ('L-pi-6', 'L-pi-7', (10,255,10)),
        ('L-pi-7', 'L-pi-8', (10,255,10)),
        ('L-pi-8', 'L-pi-9', (10,255,10)),
        ('L-pi-9', 'L-pi-10', (10,255,10)),
        ('L-pi-10', 'L-pi-11', (10,255,10)),
        ('L-pi-11', 'L-pi-12', (10,255,10)),

        ('R-pi-1', 'R-pi-2', (10,255,10)),
        ('R-pi-2', 'R-pi-3', (10,255,10)),
        ('R-pi-3', 'R-pi-4', (10,255,10)),
        ('R-pi-4', 'R-pi-5', (10,255,10)),
        ('R-pi-5', 'R-pi-6', (10,255,10)),
        ('R-pi-6', 'R-pi-7', (10,255,10)),
        ('R-pi-7', 'R-pi-8', (10,255,10)),
        ('R-pi-8', 'R-pi-9', (10,255,10)),
        ('R-pi-9', 'R-pi-10', (10,255,10)),
        ('R-pi-10', 'R-pi-11', (10,255,10)),
        ('R-pi-11', 'R-pi-12', (10,255,10)),
        
        #feijing
        ('L-fei-1', 'L-fei-2', (10,255,100)),
        ('L-fei-2', 'L-fei-3', (10,255,100)),
        ('L-fei-3', 'L-fei-4', (10,255,100)),
        ('L-fei-4', 'L-fei-5', (10,255,100)),
        ('L-fei-5', 'L-fei-6', (10,255,100)),
        ('L-fei-6', 'L-fei-7', (10,255,100)),
        ('L-fei-7', 'L-fei-8', (10,255,100)),

        ('R-fei-1', 'R-fei-2', (10,255,100)),
        ('R-fei-2', 'R-fei-3', (10,255,100)),
        ('R-fei-3', 'R-fei-4', (10,255,100)),
        ('R-fei-4', 'R-fei-5', (10,255,100)),
        ('R-fei-5', 'R-fei-6', (10,255,100)),
        ('R-fei-6', 'R-fei-7', (10,255,100)),
        ('R-fei-7', 'R-fei-8', (10,255,100))

        

    ]

COCO_PERSON_KEYPOINT_NAMES_LEFT_DOWN = ['R-sanjiao-8','R-sanjiao-9','R-sanjiao-10','R-sanjiao-11','R-sanjiao-12','R-sanjiao-13','R-sanjiao-14','R-sanjiao-15','R-sanjiao-16',
                    'R-sanjiao-17','R-sanjiao-18','R-sanjiao-19','R-sanjiao-20','R-sanjiao-21','R-sanjiao-22','R-sanjiao-23','R-sanjiao-24',
                    'R-dan-1','R-dan-2','R-dan-3','R-dan-4','R-dan-5','R-dan-6','R-dan-7','R-dan-8','R-dan-9','R-dan-10','R-dan-11','R-dan-12',
                    'R-dan-13','R-dan-14','R-dan-15','R-dan-16','R-dan-17','R-dan-18']
KEYPOINT_CONNECTION_RULES_LEFT_DOWN = [
        # sanjiao
        ('R-sanjiao-8', 'R-sanjiao-9', (255, 0 , 255)),
        ('R-sanjiao-9', 'R-sanjiao-10', (255, 0 , 255)),
        ('R-sanjiao-10', 'R-sanjiao-11', (255, 0 , 255)),
        ('R-sanjiao-11', 'R-sanjiao-12', (255, 0 , 255)),
        ('R-sanjiao-12', 'R-sanjiao-13', (255, 0 , 255)),
        ('R-sanjiao-13', 'R-sanjiao-14', (255, 0 , 255)),
        ('R-sanjiao-14', 'R-sanjiao-15', (255, 0 , 255)),
        ('R-sanjiao-15', 'R-sanjiao-16', (255, 0 , 255)),
        ('R-sanjiao-16', 'R-sanjiao-17', (255, 0 , 255)),
        ('R-sanjiao-17', 'R-sanjiao-18', (255, 0 , 255)),
        ('R-sanjiao-18', 'R-sanjiao-19', (255, 0 , 255)),
        ('R-sanjiao-19', 'R-sanjiao-20', (255, 0 , 255)),
        ('R-sanjiao-20', 'R-sanjiao-21', (255, 0 , 255)),
        ('R-sanjiao-21', 'R-sanjiao-22', (255, 0 , 255)),
        ('R-sanjiao-22', 'R-sanjiao-23', (255, 0 , 255)),
        ('R-sanjiao-23', 'R-sanjiao-24', (255, 0 , 255)),

        ('R-dan-1', 'R-dan-2', (0, 255 , 255)),
        ('R-dan-2', 'R-dan-3', (0, 255 , 255)),
        ('R-dan-3', 'R-dan-4', (0, 255 , 255)),
        ('R-dan-4', 'R-dan-5', (0, 255 , 255)),
        ('R-dan-5', 'R-dan-6', (0, 255 , 255)),
        ('R-dan-6', 'R-dan-7', (0, 255 , 255)),
        ('R-dan-7', 'R-dan-8', (0, 255 , 255)),
        ('R-dan-8', 'R-dan-9', (0, 255 , 255)),
        ('R-dan-9', 'R-dan-10', (0 ,255, 255)),
        ('R-dan-10', 'R-dan-11', (0 ,255, 255)),
        ('R-dan-11', 'R-dan-12', (0 ,255, 255)),
        ('R-dan-12', 'R-dan-13', (0 ,255, 255)),
        ('R-dan-13', 'R-dan-14', (0 ,255, 255)),
        ('R-dan-14', 'R-dan-15', (0 ,255, 255)),
        ('R-dan-15', 'R-dan-16', (0 ,255, 255)),
        ('R-dan-16', 'R-dan-17', (0 ,255, 255)),
        ('R-dan-17', 'R-dan-18', (0 ,255, 255)),

]


COCO_PERSON_KEYPOINT_NAMES_HEAD_MIDDLE_DOWN = ["R-pangguang-7","R-pangguang-30","R-pangguang-8",
                                                "L-pangguang-7","L-pangguang-30","L-pangguang-8",]

KEYPOINT_CONNECTION_RULES_HEAD_MIDDLE_DOWN = [
    ('R-pangguang-7', 'R-pangguang-30', (255, 0 , 255)),
    ('R-pangguang-30', 'R-pangguang-8', (255, 0 , 255)),

    ('L-pangguang-7', 'L-pangguang-30', (0 ,255, 255)),
    ('L-pangguang-30', 'L-pangguang-8', (0 ,255, 255)),

]


KEYPOINT_CONNECTION_RULES_WHOLE_DOWN = [
        # sanjiao
        ('R-sanjiao-1', 'R-sanjiao-2', (255, 0 , 255)),
        ('R-sanjiao-2', 'R-sanjiao-3', (255, 0 , 255)),
        ('R-sanjiao-3', 'R-sanjiao-4', (255, 0 , 255)),
        ('R-sanjiao-4', 'R-sanjiao-5', (255, 0 , 255)),
        ('R-sanjiao-5', 'R-sanjiao-6', (255, 0 , 255)),
        ('R-sanjiao-6', 'R-sanjiao-7', (255, 0 , 255)),
        ('R-sanjiao-7', 'R-sanjiao-8', (255, 0 , 255)),
        ('R-sanjiao-8', 'R-sanjiao-9', (255, 0 , 255)),

        ('L-sanjiao-1', 'L-sanjiao-2', (255, 0 , 255)),
        ('L-sanjiao-2', 'L-sanjiao-3', (255, 0 , 255)),
        ('L-sanjiao-3', 'L-sanjiao-4', (255, 0 , 255)),
        ('L-sanjiao-4', 'L-sanjiao-5', (255, 0 , 255)),
        ('L-sanjiao-5', 'L-sanjiao-6', (255, 0 , 255)),
        ('L-sanjiao-6', 'L-sanjiao-7', (255, 0 , 255)),
        ('L-sanjiao-7', 'L-sanjiao-8', (255, 0 , 255)),
        ('L-sanjiao-8', 'L-sanjiao-9', (255, 0 , 255)),
        #pangguang


        ('L-pangguang-7', 'L-pangguang-30', (0 ,255, 255)),
        ('L-pangguang-30', 'L-pangguang-8', (0 ,255, 255)),
        ('L-pangguang-8', 'L-pangguang-9', (0 ,255, 255)),
        ('L-pangguang-9', 'L-pangguang-10',  (0,255,255)),
        ('L-pangguang-10', 'L-pangguang-11', (0,255,255)),
        ('L-pangguang-11', 'L-pangguang-12', (0,255,255)),
        ('L-pangguang-12', 'L-pangguang-13', (0,255,255)),
        ('L-pangguang-13', 'L-pangguang-14', (0,255,255)),
        ('L-pangguang-14', 'L-pangguang-15', (0,255,255)),
        ('L-pangguang-15', 'L-pangguang-16', (0,255,255)),
        ('L-pangguang-16', 'L-pangguang-17', (0,255,255)),

        ('L-pangguang-18', 'L-pangguang-19', (255,255,0)),
        ('L-pangguang-19', 'L-pangguang-20', (255,255,0)),
        ('L-pangguang-20', 'L-pangguang-21', (255,255,0)),
        ('L-pangguang-21', 'L-pangguang-22', (255,255,0)),
        ('L-pangguang-22', 'L-pangguang-23', (255,255,0)),
        ('L-pangguang-23', 'L-pangguang-24', (255,255,0)),


        ('R-pangguang-7', 'R-pangguang-30', (0,255,255)),
        ('R-pangguang-30', 'R-pangguang-8', (0,255,255)),
        ('R-pangguang-8', 'R-pangguang-9', (0,255,255)),
        ('R-pangguang-9', 'R-pangguang-10',  (0,255,255)),
        ('R-pangguang-10', 'R-pangguang-11', (0,255,255)),
        ('R-pangguang-11', 'R-pangguang-12', (0,255,255)),
        ('R-pangguang-12', 'R-pangguang-13', (0,255,255)),
        ('R-pangguang-13', 'R-pangguang-14', (0,255,255)),
        ('R-pangguang-14', 'R-pangguang-15', (0,255,255)),
        ('R-pangguang-15', 'R-pangguang-16', (0,255,255)),
        ('R-pangguang-16', 'R-pangguang-17', (0,255,255)),

        ('R-pangguang-18', 'R-pangguang-19', (255,255,0)),
        ('R-pangguang-19', 'R-pangguang-20', (255,255,0)),
        ('R-pangguang-20', 'R-pangguang-21', (255,255,0)),
        ('R-pangguang-21', 'R-pangguang-22', (255,255,0)),
        ('R-pangguang-22', 'R-pangguang-23', (255,255,0)),
        ('R-pangguang-23', 'R-pangguang-24', (255,255,0)),    
    ]






