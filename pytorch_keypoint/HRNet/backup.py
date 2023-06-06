#*********************************************
'''
origin_w ,origin_h = target["image_width"],target["image_height"]
        result_img = np.zeros((1024,512,3),dtype = np.uint8)
        pad_left_w = (512-origin_w)//2
        pad_right_w = 512-origin_w-pad_left_w
        pad_top_h = (1024-origin_h) // 2
        pad_down_h = 1024 - origin_h - pad_top_h

        target["keypoints"][:,0] += pad_left_w
        target["keypoints"][:,1] += pad_top_h
        result_img[pad_top_h:pad_top_h + origin_h, pad_left_w:pad_left_w + origin_w,:] = image

'''

