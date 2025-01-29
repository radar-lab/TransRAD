#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lei
"""
import numpy as np
import utils.radar_loader as loader
import utils.helper as helper
from utils.utils import resize_radar_data

####
def loadDataForPlot(all_RAD_files, config_data, config_inference, \
                    config_radar,dataset_dir="test_set_dir", interpolation=15.):
    """ Load data one by one for generating evaluation images """
    sequence_num = -1
    for RAD_file in [all_RAD_files[0]]:#all_RAD_files
        sequence_num += 1
        ### load RAD input ###
        RAD_complex = loader.readRAD(RAD_file)

        ### NOTE: real time visualization ###
        RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                        power_order=2), target_axis=-1), scalar=10, log_10=True)
        RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                        power_order=2), target_axis=1), scalar=10, log_10=True)
        RA_cart = helper.toCartesianMask(RA, config_radar, \
                                gapfill_interval_num=int(interpolation))
        RA_img = helper.norm2Image(RA)[..., :3]
        RD_img = helper.norm2Image(RD)[..., :3]
        RA_cart_img = helper.norm2Image(RA_cart)[..., :3]

        img_file = loader.imgfileFromRADfile(RAD_file, config_data[dataset_dir])
        stereo_left_image = loader.readStereoLeft(img_file)

        RAD_data = helper.complexTo2Channels(RAD_complex)
        #   进行归一化, 并添加上batch_size维度
        RAD_data = (RAD_data - config_data["global_mean_log"]) / \
                            config_data["global_variance_log"]
        #data = tf.expand_dims(tf.constant(RAD_data, dtype=tf.float32), axis=0)
        data = np.expand_dims(RAD_data, axis=0)
        yield sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img
    


####
def RAD_RA_RD_AD_for_one_frame(RAD_sequences,frame,config_data,dataset_dir="train_set_dir"):##test--->"test_set_dir"
    """ Parse RAD data """
    #count = 0
    #while  count < len(self.RAD_sequences_train):
    RAD_filename = RAD_sequences[frame] 
    ## load radar RAD
    RAD_complex = loader.readRAD(RAD_filename)
    if RAD_complex is None:
        raise ValueError("RAD file not found, please double check the path")
    ### NOTE: Gloabl Normalization ###
    RAD_data = helper.complexTo2Channels(RAD_complex)
    RAD_data = (RAD_data - config_data["global_mean_log"]) / \
                        config_data["global_variance_log"]
    ### load ground truth instances ###
    gt_filename = loader.gtfileFromRADfile(RAD_filename, \
                                config_data[dataset_dir])
    gt_instances = loader.readRadarInstances(gt_filename)#for the boxes, the format is [obj_number,x_center, y_center, z_center, w, h, d].for the cart_box, the format is [obj_number,x_center, y_center, w, h].
    if gt_instances is None:
        raise ValueError("gt file not found, please double check the path")
        
    ### initialize gronud truth raw_boxes as np.zeors ### [obj_number,x_center, y_center, z_center, w, h, d,class_id]
    raw_boxes_xyzwhd = np.zeros((config_data["max_boxes_per_frame"], 7))
    ### start getting ground turth raw_boxes ###
    for i in range(len(gt_instances["classes"])):
        if i > config_data["max_boxes_per_frame"]:
            continue
        class_name = gt_instances["classes"][i]
        box_xyzwhd = gt_instances["boxes"][i]
        class_id = config_data["all_classes"].index(class_name)
        if i < config_data["max_boxes_per_frame"]:
            raw_boxes_xyzwhd[i, :6] = box_xyzwhd
            raw_boxes_xyzwhd[i, 6] = class_id
            
    ### Transform RAD data to RA, RD, AD ###(x-0, y-1, z-2, w-3, h-4, d-5,class_id-6)-->(0,1,2,3,4,5,6)
    RAD_label = raw_boxes_xyzwhd
    
    RA              =np.mean(RAD_data, axis=2) #shape-->[R,A]   #np.sum(RAD, axis=2)
    RA_retained_idxs=(0,1,3,4,6) #(x, y, w, h, class_id)
    RA_label        =RAD_label[:,RA_retained_idxs] #shape-->[obj_num,(x, y, w, h, class_id)]
    
    RD              =np.mean(RAD_data, axis=1) #shape-->[R,D]
    RD_retained_idxs=(0,2,3,5,6)
    RD_label        =RAD_label[:,RD_retained_idxs] #shape-->[obj_num,(x, z, w, d, class_id)]
    
    AD              =np.mean(RAD_data, axis=0) #shape-->[A,D]
    AD_retained_idxs=(1,2,4,5,6)
    AD_label        =RAD_label[:,AD_retained_idxs] #shape-->[obj_num,(y, z, h, d, class_id)]
    
    return RAD_data, RAD_label,RA, RA_label, RD, RD_label, AD, AD_label

####
def RADData_parse_for_one_frame(RAD_sequences,frame_idx,config_data,dataset_dir="train_set_dir"):##test--->"test_set_dir"
    """ Parse RAD data """
    #count = 0
    #while  count < len(self.RAD_sequences_train):
    RAD_filename = RAD_sequences[frame_idx] 
    ## load radar RAD
    RAD_complex = loader.readRAD(RAD_filename)
    if RAD_complex is None:
        raise ValueError("RAD file not found, please double check the path")
    ### NOTE: Gloabl Normalization ###
    RAD_data = helper.complexTo2Channels(RAD_complex)
    RAD_data = (RAD_data - config_data["global_mean_log"]) / \
                        config_data["global_variance_log"]
    ### load ground truth instances ###
    gt_filename = loader.gtfileFromRADfile(RAD_filename, \
                                config_data[dataset_dir])
    gt_instances = loader.readRadarInstances(gt_filename)#for the boxes, the format is [obj_number,x_center, y_center, z_center, w, h, d].for the cart_box, the format is [obj_number,x_center, y_center, w, h].
    if gt_instances is None:
        raise ValueError("gt file not found, please double check the path")
        
    ### initialize gronud truth raw_boxes as np.zeors ###
    #raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
    raw_boxes_xyzwhd = np.zeros((len(gt_instances["classes"]), 7))
    ### start getting ground turth raw_boxes ###
    for i in range(len(gt_instances["classes"])):
        if i > config_data["max_boxes_per_frame"]:
            continue
        class_name = gt_instances["classes"][i]
        box_xyzwhd = gt_instances["boxes"][i]
        class_id = config_data["all_classes"].index(class_name)
        if i < config_data["max_boxes_per_frame"]:
            raw_boxes_xyzwhd[i, :6] = box_xyzwhd
            raw_boxes_xyzwhd[i, 6] = class_id
            
    ### NOTE: decode ground truth boxes to YOLO format ###
    #gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

    #if has_label:
    return np.array(RAD_data, dtype=np.float32), np.array(raw_boxes_xyzwhd, dtype=np.float32)



### Excerpt from radar_dataset_generator.py   
def preprocess_RAD_Label(image, box, input_shape, mode = 'RA', test= True):
    ''' mode= 'RAD','RA', 'RD', 'AD' '''
    # image       = np.array(RAD_data, dtype=np.float32) #[H, W, C] or [H, W, D] or [R, A, D]
    # box         = np.array(raw_boxes_xyzwhd, dtype=np.float32)
    #---------------------------------------------------#
    # GT_labels = icxyzwhd;  box = xyzwhdc
    #---------------------------------------------------#
    nL          = len(box)
    labels_out  = np.zeros((nL, 8)) #6-icxywh; 8-icxyzwhd
    GT_labels   = labels_out
    if nL: 

        if not test:
            #---------------------------------------------------#
            # box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1] #[x1, x2]/ w
            # box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0] #[y1, y2]/ h
            box[:, [0,3]] = box[:, [0,3]] / input_shape[1] # x,w / w0
            box[:, [1,4]] = box[:, [1,4]] / input_shape[0] # y,h / h0
            box[:, [2,5]] = box[:, [2,5]] / input_shape[2] # z,c / c0

        #---------------------------------------------------#
        labels_out[:, 1]  = box[:, -1]
        labels_out[:, 2:] = box[:, :-1]
        #####################################################
        #####################################################
        if mode == 'RAD':  # R-H, A-W, D-D
            ### 5-dimensional tensor in PyTorch is [B,C,D,H,W] for 3D convolutions
            # [H, W, D] -> [C, H, W, D]
            image       = np.expand_dims(image, axis=0)
            # Rearrange [C, H, W, D] to [C, D, H, W]
            image       = np.transpose(image, (0, 3, 1, 2))
            GT_labels   = labels_out # [img_idx,class,xyz,whd]= [i, c, x, y, z, w, h, d] 
        elif mode == 'RA':
            ### 4-dimensional tensor in PyTorch is [B,C,H,W] for 2D convolutions
            # convert from [H, W, D] to [D, H, W] with D as Channel for pytorch
            image       = np.transpose(image, (2, 0, 1)) #RA 
            # Extracting array with shape [i, c, x, y, w, h]
            GT_labels = labels_out[:, [0, 1, 2, 3, 5, 6]]
        elif mode == 'RD':
            # convert from [H, W, D] to [W, H, D] with W as Channel for pytorch
            image       = np.transpose(image, (1, 0, 2))                  
            # Extracting array with shape [i, c, x, z, w, d]
            GT_labels = labels_out[:, [0, 1, 2, 4, 5, 7]]
        elif mode == 'AD':
            # convert from [H, W, D] to [H, W, D] with H as Channel for pytorch
            image       = np.transpose(image, (0, 1, 2))                  
            # Extracting array with shape [i, c, y, z, h, d]
            GT_labels = labels_out[:, [0, 1, 3, 4, 6, 7]] 
            
    if test:  #  icxyzwhd ---> ixyzwhdc
        GT_labels = np.hstack((GT_labels[:, :1], GT_labels[:, 2:], GT_labels[:, 1:2]))
        
    return image, GT_labels


def getitem_RADframe(RAD_sequences,frame_idx,config_data, input_shape, target_shape, interpolation, mode, dataset_dir="train_set_dir"):
    RAD_data, raw_boxes_xyzwhd = RADData_parse_for_one_frame(RAD_sequences,frame_idx,config_data,dataset_dir)
    RAD_data             = resize_radar_data(RAD_data, target_shape, interpolation)

    RADframe, labels_out = preprocess_RAD_Label(RAD_data, raw_boxes_xyzwhd, input_shape, mode)
    del RAD_data    
    return RADframe, labels_out
##########################################
if __name__ == "__main__":
    #----------------------------------------------------#
    #   Load config parameters with using .json
    #----------------------------------------------------#    
    config = loader.readConfig()
    config_data = config["DATA"]
    # config_radar = config["RADAR_CONFIGURATION"]
    # config_model = config["MODEL"]
    # config_train = config["TRAIN"]
    
    ### radar data reading
    RAD_sequences_train = loader.readSequences(config_data,mode="train")
    RAD_sequences_test = loader.readSequences(config_data,mode="test")
    # RAD_sequences_train, RAD_sequences_validate = loader.splitTrain(RAD_sequences_train,validate=True)
    # num_val     = len(RAD_sequences_validate) 
    num_train   = len(RAD_sequences_train)
    num_test    = len(RAD_sequences_test)
    
    ### initialize array ### R,A,D--->x-w,y-h,z-d;   R--->x-w;  A--->y-h;  D--->z-d
    RAD      =[] #shape-->[frame,R,A,D]
    RAD_label=[] #shape-->[frame,obj_num,(x, y, z, w, h, d,class_id)]-->[frame,obj_num,7]
    
    RA       =[] #shape-->[frame,R,A]
    RA_label =[] #shape-->[frame,obj_num,(x, y, w, h, class_id)]
    
    RD       =[] #shape-->[frame,R,D]
    RD_label =[] #shape-->[frame,obj_num,(x, z, w, d, class_id)]
    
    AD       =[] #shape-->[frame,A,D]
    AD_label =[] #shape-->[frame,obj_num,(y, z, h, d, class_id)]
    
    ### Accumulate RAD data and label
    for frame in range(3):
    #for frame in range(len(RAD_sequences_train)):
        RAD_frame, RAD_label_frame = RADData_parse_for_one_frame(RAD_sequences_train,frame,config_data,dataset_dir="train_set_dir")
        RAD.append(RAD_frame)
        RAD_label.append(RAD_label_frame)
    ###
    RAD=np.array(RAD)
    RAD_label=np.array(RAD_label)
    ### Transform RAD data to RA, RD, AD ###(x-0, y-1, z-2, w-3, h-4, d-5,class_id-6)-->(0,1,2,3,4,5,6)
    RA              =np.mean(RAD, axis=3) #shape-->[frame,R,A]   #np.sum(RAD, axis=3)
    RA_retained_idxs=(0,1,3,4,6) #(x, y, w, h, class_id)
    RA_label        =RAD_label[:,:,RA_retained_idxs] #shape-->[frame,obj_num,(x, y, w, h, class_id)]
    
    RD              =np.mean(RAD, axis=2) #shape-->[frame,R,D]
    RD_retained_idxs=(0,2,3,5,6)
    RD_label        =RAD_label[:,:,RD_retained_idxs] #shape-->[frame,obj_num,(x, z, w, d, class_id)]
    
    AD              =np.mean(RAD, axis=1) #shape-->[frame,A,D]
    AD_retained_idxs=(1,2,4,5,6)
    AD_label        =RAD_label[:,:,AD_retained_idxs] #shape-->[frame,obj_num,(y, z, h, d, class_id)]
    
 
    ### Transform RAD data to RA, RD, AD for one frame directly
    RAD_data, RAD_label,RA, RA_label, RD, RD_label, AD, AD_label = RAD_RA_RD_AD_for_one_frame(RAD_sequences_train,0,config_data,dataset_dir="train_set_dir")