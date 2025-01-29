from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input,resize_radar_data
## radar-Lei
import utils.radar_loader as loader
import utils.helper as helper
from nets.yolo_training import xyxy2xywh,xyxy2xywh,xyzwhd2xyzxyz,xyzxyz2xyzwhd

class RadarDataset(Dataset):
    def __init__(self, RAD_sequences, config_data, dataset_dir, input_shape, num_classes, epoch_length, \
                       train, mode = 'RA', target_shape=[256,256,256], interpolation='nearest', test=False):
        """ 
        Generate batch size of train data (Img and GT_BBox)
        mode= 'RAD','RA', 'RD', 'AD'
        """
        super(RadarDataset, self).__init__()
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.epoch_length       = epoch_length
        self.train              = train
        self.test               = test
        self.epoch_now          = -1
        
        ## radar
        self.RAD_sequences      = RAD_sequences #annotation_lines
        self.length             = len(self.RAD_sequences)
        self.config_data        = config_data
        self.mode               = mode
        self.dataset_dir        = dataset_dir 
        self.target_shape       =target_shape
        self.interpolation      =interpolation

    def __len__(self):
        return self.length

        
    def clamp_box(self,arr, y_shape, z_shape):
        """
        Adjusts y1, y2, z1, and z2 coordinates based on the provided logic.
        
        Parameters:
        arr (np.ndarray): Input array with shape [N, 6] where columns represent [x1, y1, z1, x2, y2, z2]
        y_shape (int): The shape parameter for y coordinates
        z_shape (int): The shape parameter for z coordinates
        
        Returns:
        np.ndarray: Adjusted array
        """
        arr=xyzwhd2xyzxyz(arr)
        # Iterate over each row in the array
        for i in range(arr.shape[0]):
            x1, y1, z1, x2, y2, z2 = arr[i]
            
            # Adjust y1 and y2
            if y1 < 0:
                y1 += y_shape
                if (y_shape - y1) >= (y2 - 0):
                    y1, y2 = y1, y_shape
                else:
                    y1, y2 = 0, y2
            elif y2 >= y_shape:
                y2 -= y_shape
                if (y_shape - y1) >= (y2 - 0):
                    y1, y2 = y1, y_shape
                else:
                    y1, y2 = 0, y2
            
            # Adjust z1 and z2
            if z1 < 0:
                z1 += z_shape
                if (z_shape - z1) >= (z2 - 0):
                    z1, z2 = z1, z_shape
                else:
                    z1, z2 = 0, z2
            elif z2 >= z_shape:
                z2 -= z_shape
                if (z_shape - z1) >= (z2 - 0):
                    z1, z2 = z1, z_shape
                else:
                    z1, z2 = 0, z2
            
            # Update the array with adjusted coordinates
            arr[i] = [x1, y1, z1, x2, y2, z2]
            
        arr = xyzxyz2xyzwhd(arr)
        
        return arr 
    
    def clamp_box2(self,arr, y_shape, z_shape):
        """
        Adjusts y1, y2, z1, and z2 coordinates based on the provided logic.
        
        Parameters:
        arr (np.ndarray): Input array with shape [N, 6] where columns represent [x1, y1, z1, x2, y2, z2]
        y_shape (int): The shape parameter for y coordinates
        z_shape (int): The shape parameter for z coordinates
        
        Returns:
        np.ndarray: Adjusted array
        """
        arr=xyzwhd2xyzxyz(arr)
        # Create copies of y1, y2, z1, and z2 to avoid modifying the original array
        y1 = arr[:, 1].copy()
        y2 = arr[:, 4].copy()
        z1 = arr[:, 2].copy()
        z2 = arr[:, 5].copy()
        
        # Adjust y1 and y2
        y1_neg_mask = y1 < 0
        y2_ge_shape_mask = y2 >= y_shape
        
        y1[y1_neg_mask] += y_shape
        y2[y2_ge_shape_mask] -= y_shape
        
        y1_adjust_mask = (y_shape - y1) >= y2
        y1_new = np.where(y1_adjust_mask, y1, 0)
        y2_new = np.where(y1_adjust_mask, y_shape, y2)
        
        y1 = np.where(y1_neg_mask | y2_ge_shape_mask, y1_new, y1)
        y2 = np.where(y1_neg_mask | y2_ge_shape_mask, y2_new, y2)
        
        # Adjust z1 and z2
        z1_neg_mask = z1 < 0
        z2_ge_shape_mask = z2 >= z_shape
        
        z1[z1_neg_mask] += z_shape
        z2[z2_ge_shape_mask] -= z_shape
        
        z1_adjust_mask = (z_shape - z1) >= z2
        z1_new = np.where(z1_adjust_mask, z1, 0)
        z2_new = np.where(z1_adjust_mask, z_shape, z2)
        
        z1 = np.where(z1_neg_mask | z2_ge_shape_mask, z1_new, z1)
        z2 = np.where(z1_neg_mask | z2_ge_shape_mask, z2_new, z2)
        
        # Update the array with adjusted coordinates
        arr[:, 1] = y1
        arr[:, 4] = y2
        arr[:, 2] = z1
        arr[:, 5] = z2
        
        arr = xyzxyz2xyzwhd(arr)
        return arr    
    ############  RADData Loader  #####################################
    def RADData_generator(self,frame_idx):
        """
        Generate train data(Img and GT_BBox) with batch size
        """
        #count = 0
        #while  count < len(self.RAD_sequences_train):
        RAD_filename = self.RAD_sequences[frame_idx] 
        ## load radar RAD
        RAD_complex = loader.readRAD(RAD_filename)
        if RAD_complex is None:
            raise ValueError("RAD file not found, please double check the path")
        ### NOTE: Gloabl Normalization ###
        RAD_data = helper.complexTo2Channels(RAD_complex)
        RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                            self.config_data["global_variance_log"]
        ### load ground truth instances ###
        gt_filename = loader.gtfileFromRADfile(RAD_filename, \
                                    self.config_data[self.dataset_dir])
        gt_instances = loader.readRadarInstances(gt_filename)#for the boxes, the format is [obj_number,x_center, y_center, z_center, w, h, d].for the cart_box, the format is [obj_number,x_center, y_center, w, h].
        if gt_instances is None:
            raise ValueError("gt file not found, please double check the path")
            
        ### initialize gronud truth raw_boxes as np.zeors ###
        #raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        raw_boxes_xyzwhd = np.zeros((len(gt_instances["classes"]), 7))
        ### start getting ground turth raw_boxes ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
                
        ### NOTE: decode ground truth boxes to YOLO format ###
        #gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)

        #if has_label:
        return np.array(RAD_data, dtype=np.float32), np.array(raw_boxes_xyzwhd, dtype=np.float32)
        # count += 1
        # if count == len(self.RAD_sequences_train) - 1:
        #     # np.random.seed() # should I add seed here ?
        #     np.random.shuffle(self.RAD_sequences_train)   

    def preprocess_RAD_Label(self, image, box):
        ''' mode= 'RAD','RA', 'RD', 'AD' '''
        # image       = np.array(RAD_data, dtype=np.float32) #[H, W, C] or [H, W, D] or [R, A, D]
        # box         = np.array(raw_boxes_xyzwhd, dtype=np.float32)

        nL          = len(box)
        labels_out  = np.zeros((nL, 8)) #6-ixywhc; 8-ixyzwhdc
        GT_labels   = labels_out
        if nL:

            if not self.test:

                #---------------------------------------------------#
                # box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1] #[x1, x2]/ w
                # box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0] #[y1, y2]/ h
                box[:, [0,3]] = box[:, [0,3]] / self.input_shape[0] # x,w / w0
                box[:, [1,4]] = box[:, [1,4]] / self.input_shape[1] # y,h / h0
                box[:, [2,5]] = box[:, [2,5]] / self.input_shape[2] # z,c / c0

            #---------------------------------------------------#
            labels_out[:, 1]  = box[:, -1]
            labels_out[:, 2:] = box[:, :-1]
            #####################################################
            #####################################################
            if self.mode == 'RAD':  # R-H, A-W, D-D
                ### 5-dimensional tensor in PyTorch is [B,C,D,H,W] for 3D convolutions
                # [H, W, D] -> [C, H, W, D]
                image       = np.expand_dims(image, axis=0)
                # Rearrange [C, H, W, D] to [C, D, H, W]
                image       = np.transpose(image, (0, 3, 1, 2))
                GT_labels   = labels_out # [img_idx,class,xyz,whd]= [i, c, x, y, z, w, h, d] 
            elif self.mode == 'RA':
                ### 4-dimensional tensor in PyTorch is [B,C,H,W] for 2D convolutions
                # convert from [H, W, D] to [D, H, W] with D as Channel for pytorch
                image       = np.transpose(image, (2, 0, 1)) #RA 
                # Extracting array with shape [i, c, x, y, w, h]
                #GT_labels = labels_out[:, [0, 1, 2, 3, 5, 6]]
                GT_labels   = labels_out # [img_idx,class,xyz,whd]= [i, c, x, y, z, w, h, d]
            elif self.mode == 'RD':
                # convert from [H, W, D] to [W, H, D] with W as Channel for pytorch
                image       = np.transpose(image, (1, 0, 2))                  
                # Extracting array with shape [i, c, x, z, w, d]
                GT_labels = labels_out[:, [0, 1, 2, 4, 5, 7]]
            elif self.mode == 'AD':
                # convert from [H, W, D] to [H, W, D] with H as Channel for pytorch
                image       = np.transpose(image, (0, 1, 2))                  
                # Extracting array with shape [i, c, y, z, h, d]
                GT_labels = labels_out[:, [0, 1, 3, 4, 6, 7]]
            
        if self.test:  #  icxyzwhd ---> ixyzwhdc
            GT_labels = np.hstack((GT_labels[:, :1], GT_labels[:, 2:], GT_labels[:, 1:2]))
            
        return image, GT_labels

    def __getitem__(self, index):
        index       = index % self.length


        #image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        RAD_data, raw_boxes_xyzwhd = self.RADData_generator(index) #[H, W, C]
        xyzwhd = self.clamp_box(raw_boxes_xyzwhd[...,:6], self.input_shape[1], self.input_shape[2])
        raw_boxes_xyzwhd[..., :6] = xyzwhd
        RAD_data             = resize_radar_data(RAD_data, self.target_shape, self.interpolation)

        image, labels_out = self.preprocess_RAD_Label(RAD_data, raw_boxes_xyzwhd)
        del RAD_data    
        return image, labels_out

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    

def radar_dataset_collate(batch):
    images  = []
    bboxes  = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes


