# Authors: Lei Cheng, Ao Zhang, Erlik Nowruzi, Robert Laganiere
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from glob import glob
from tqdm import tqdm
from time import sleep
from tabulate import tabulate
from functools import partial
from datetime import datetime
import torch
from torch.utils.data import DataLoader


import utils.radar_loader as loader
import utils.helper as helper
import utils.drawer as drawer
from utils.utils import get_classes,show_config, worker_init_fn

import mAP as mAP
from yolo import YOLO
from utils.radar_dataset_generator import RadarDataset, radar_dataset_collate
from Load_radar_data import getitem_RADframe,loadDataForPlot,RADData_parse_for_one_frame
import time

inference_time = []
Plot_result  = True  #False 


def Plot_Bbox(image_save_dir,RAD_mag, nms_pred, frame_idx,RAD_sequences,
              stereo_image_test,gt_bboxes, config_data):
    ### load RAD input ###
    RAD_filename = RAD_sequences[frame_idx]
    RAD_complex = loader.readRAD(RAD_filename)   
    ### NOTE: real time visualization ###
    RA = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                    power_order=2), target_axis=-1), scalar=10, log_10=True)
    RD = helper.getLog(helper.getSumDim(helper.getMagnitude(RAD_complex, \
                    power_order=2), target_axis=1), scalar=10, log_10=True)    
    RA_img = helper.norm2Image(RA)[..., :3]
    RD_img = helper.norm2Image(RD)[..., :3]
    #RA_cart_img = helper.norm2Image(RA_cart)[..., :3]
    RA_cart_img =  None
    ### load ground truth instances ###
    gt_filename  = loader.gtfileFromRADfile(RAD_filename, config_data["test_set_dir"])
    gt_instances = loader.readRadarInstances(gt_filename)#for the boxes, the format is [obj_number,x_center, y_center, z_center, w, h, d].for the cart_box, the format is [obj_number,x_center, y_center, w, h].

    
    img_filename = stereo_image_test[frame_idx]
    stereo_left_image = loader.readStereoLeft(img_filename)
    #################    Plot for each frame   ###########################################
    if_evaluate_cart = False
    if if_evaluate_cart:
        fig, axes = drawer.prepareFigure(4, figsize=(25, 8))
    else:
        fig, axes = drawer.prepareFigure(3, figsize=(16, 6))
    #fig, axes = drawer.prepareFigure(2, figsize=(15, 6))
    colors = loader.randomColors(config_data["all_classes"])

    
    ## draw bbox for without_bbox 
    drawer.clearAxes(axes)
    drawer.drawInference(stereo_left_image, RD_img, RA_img, RA_cart_img, nms_pred, \
                    config_data["all_classes"], colors, axes, \
                    radar_cart_nms=None,draw_RD=True,draw_no_box=True)              
    drawer.saveFigure(image_save_dir+'no_box/', "%.6d.png"%(frame_idx)) 
    
    ## draw bbox for gt 
    drawer.clearAxes(axes)
    drawer.drawInference(stereo_left_image, RD_img, RA_img, RA_cart_img, gt_bboxes, \
                    config_data["all_classes"], colors, axes, \
                    radar_cart_nms=None,draw_RD=True,for_gt=True)              
    drawer.saveFigure(image_save_dir+'gt/', "%.6d.png"%(frame_idx)) 
    
    ## draw bbox for det 
    drawer.clearAxes(axes)
    drawer.drawInference(stereo_left_image, RD_img, RA_img, RA_cart_img, nms_pred, \
                    config_data["all_classes"], colors, axes, \
                    radar_cart_nms=None,draw_RD=True)              
    drawer.saveFigure(image_save_dir+'det/', "%.6d.png"%(frame_idx)) 
    
    ## draw bbox for with_gt 
    drawer.clearAxes(axes)
    drawer.drawRadarPredWithGt(stereo_left_image, RD_img, RA_img, RA_cart_img, \
                            gt_instances, nms_pred, config_data["all_classes"], colors, axes, \
                            radar_cart_nms=None)    
    drawer.saveFigure(image_save_dir+'with_gt/', "%.6d.png"%(frame_idx)) 
    print("\n Save one image \n")
    #######################################################################################
 
    
    

def compute_mAP(yolo, gen, map_iou_threshold_list, cuda, num_classes, 
                config_model,batch_size, total_test_batches,RAD_sequences_test,\
                stereo_image_test, config_data,mode='RA'):
    mean_ap_test_all = [] 
    ap_all_class_test_all = []
    ap_all_class_all = []
    for i in range(len(map_iou_threshold_list)):
        mean_ap_test_all.append(0.0)
        ap_all_class_test_all.append([])
        ap_all_class = []
        for class_id in range(num_classes):
            ap_all_class.append([])
        ap_all_class_all.append(ap_all_class)
        
    print("Start evaluating RAD Boxes on the entire dataset, it might take a while...")
    pbar = tqdm(total=int(total_test_batches))
    for iteration, batch in enumerate(gen):
        ### Processing one batch
        if iteration >= total_test_batches:
            break
        ## gt_bboxes = ixyzwhdc
        data, gt_bboxes = batch[0], batch[1]
        if mode=='RA':
            gt_bboxes = gt_bboxes[:, [0, 1, 2, 4, 5, 7]]
        elif mode=='RD':
            gt_bboxes = gt_bboxes[:, [0, 1, 3, 4, 6, 7]] #RD
        with torch.no_grad():
            if cuda:
                data = data.cuda()
                gt_bboxes = gt_bboxes.cuda()
        for frame_id in range(data.shape[0]):
            gt_bboxes_frame = gt_bboxes[gt_bboxes[:, 0] == frame_id][:, 1:] #xyzwhdc
            
            RAD_mag = data[frame_id].cpu().detach().numpy()
            inference_time_start = time.time()            
            nms_pred,r_image = yolo.detect_image(RAD_mag , mode ) 
            inference_time.append(time.time() - inference_time_start)
            print("\n Num of GT and Detections:", len(gt_bboxes_frame), "<->", len(nms_pred),"\n")
            if Plot_result:
                #################    Plot for each frame   ###########################################
                image_save_dir = "./images/inference_plots/"
                frame_idx = (iteration * data.shape[0]) + frame_id
                Plot_Bbox(image_save_dir,RAD_mag, nms_pred, frame_idx,RAD_sequences_test,\
                          stereo_image_test,gt_bboxes_frame.cpu().detach().numpy(), config_data)                        
                ###########################################
            
            if mode=='RA':
                nms_pred = nms_pred[:, [0, 1, 3, 4, 6, 7]]
            elif mode=='RD':
                nms_pred = nms_pred[:, [0, 2, 3, 5, 6, 7]] #RD            
            for j in range(len(map_iou_threshold_list)):
                map_iou_threshold = map_iou_threshold_list[j]
                if mode=='RAD':
                    mean_ap, ap_all_class_all[j] = mAP.mAP(nms_pred, \
                                                    gt_bboxes_frame.cpu().detach().numpy(), \
                                                    config_model["input_shape"], \
                                                    ap_all_class_all[j], \
                                                    tp_iou_threshold=map_iou_threshold)
                else:
                    mean_ap, ap_all_class_all[j] = mAP.mAP2D(nms_pred, \
                                                    gt_bboxes_frame.cpu().detach().numpy(), \
                                                    config_model["input_shape"], \
                                                    ap_all_class_all[j], \
                                                    tp_iou_threshold=map_iou_threshold)
                mean_ap_test_all[j] += mean_ap                

        pbar.update(1)
    for iou_threshold_i in range(len(map_iou_threshold_list)):
        ap_all_class = ap_all_class_all[iou_threshold_i]
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_all[iou_threshold_i].append(class_ap)
        mean_ap_test_all[iou_threshold_i] /= batch_size*total_test_batches
    return mean_ap_test_all, ap_all_class_test_all


def main():
    
    #------------------------------------------------------#
    #   Output Folder
    #------------------------------------------------------#    
    image_save_dir = "./lei_outputs/Test_plots/"
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)
    else:
        shutil.rmtree(image_save_dir)
        os.makedirs(image_save_dir)

    Test_save_dir = "./Test_results/"
    if not os.path.exists(Test_save_dir):
        os.makedirs(Test_save_dir)
   
    #------------------------------------------------------#
    #   Global Parameters
    #------------------------------------------------------#
    RAD_mode        = 'RA'
    input_shape     = [256,256,64]
    target_shape    = [256,256,256]
    interpolation   = 'nearest'
    anchors_path    = "model_data/RAD_anchors.txt"
    classes_path    = 'model_data/radar_classes.txt'
    Init_Epoch          = 0
    UnFreeze_Epoch      = 50
    batch_size          = 12 #8
    #grid_strides =  [32,16,8]
    mAP_iou3d_threshold= [0.1, 0.3, 0.4, 0.5, 0.6, 0.7]
    mAP_iou2d_threshold= [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mAP_COCO_threshold= [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95] #mAP@[0.5:0.05:0.95]
 
    class_names, num_classes = get_classes(classes_path)

    #------------------------------------------------------#
    #   Load Configures
    #------------------------------------------------------#
    config = loader.readConfig()
    config_data  = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_inference = config["INFERENCE"]
    config_evaluate = config["EVALUATE"]
    #------------------------------------------------------#
    #   Preparing data: split train and validation
    #------------------------------------------------------#
    RAD_sequences_train = loader.readSequences(config_data,mode="train")
    RAD_sequences_test = loader.readSequences(config_data,mode="test")
    RAD_sequences_test = sorted(RAD_sequences_test)
    RAD_sequences_train, RAD_sequences_validate = loader.splitTrain(RAD_sequences_train,validate=True)
    num_train   = len(RAD_sequences_train)
    num_val     = len(RAD_sequences_validate) 
    num_test    = len(RAD_sequences_test)    
    total_test_batches    =  num_test//batch_size
    
    stereo_image_test = loader.readSequences(config_data,mode="test", base = "stereo_image", pattern = "/*/*.jpg")
    stereo_image_test = sorted(stereo_image_test)

    train_annotation_path   = "train_set_dir" #config_data["train_set_dir"]
    val_annotation_path     = train_annotation_path
    test_annotation_path    = "test_set_dir" #config_data["test_set_dir"]

    
    #------------------------------------------------------#
    #   Loading data
    #------------------------------------------------------#

    # ###test dataset
    test_dataset = RadarDataset(RAD_sequences_test, config_data, test_annotation_path, input_shape, 
                                num_classes, UnFreeze_Epoch,train=False, mode=RAD_mode, target_shape=target_shape, 
                                interpolation=interpolation, test=True)
    gen_test     = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, num_workers = 0, pin_memory=True, 
                                drop_last=True, collate_fn=radar_dataset_collate, sampler=None, 
                                worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    

    yolo = YOLO() 
    
    #------------------------------------------------------#
    #   Evaluation starting from here
    #------------------------------------------------------#
    # mode='RAD'
    # mAP_iou_threshold = mAP_iou3d_threshold
    mode='RAD'  #'RA' 'RD'
    if mode == 'RAD':
        mAP_iou_threshold = mAP_iou3d_threshold
    else:
        mAP_iou_threshold = mAP_iou2d_threshold
    all_mean_aps, all_ap_classes = compute_mAP(yolo, gen_test, mAP_iou_threshold, yolo.cuda, num_classes, 
                                               config_model,batch_size, total_test_batches,RAD_sequences_test,stereo_image_test,config_data, mode)

    
    ### NOTE: Inference Time ###
    print(" The RAD Boxes inference time for one frame: ", \
                                                np.mean(inference_time))    
    
    ### NOTE: evaluate RAD Boxes under different mAP_iou ###
    all_mean_aps = np.array(all_mean_aps)
    all_ap_classes = np.array(all_ap_classes)
    print('all_mean_aps: ',all_mean_aps)
    table = []
    row = []
    for i in range(len(all_mean_aps)):
        if i == 0:
            row.append("mAP")
        row.append(all_mean_aps[i])
    table.append(row)
    row = []
    for j in range(all_ap_classes.shape[1]):
        ap_current_class = all_ap_classes[:, j]
        for k in range(len(ap_current_class)):
            if k == 0:
                row.append("AP_" + config_data["all_classes"][j])
            row.append(ap_current_class[k])
        table.append(row)
        row = []
    headers = []
    for ap_iou_i in mAP_iou_threshold:
        if ap_iou_i == 0:
            headers.append("AP name")
        headers.append("AP_%.2f"%(ap_iou_i))
    print("==================== RAD Boxes AP ========================")
    print(tabulate(table, headers=headers))
    print("==========================================================")
    

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'rad_boxes_ap_{current_time}.txt'
    with open(os.path.join(Test_save_dir,file_name), 'w') as f:
        f.write("==================== RAD Boxes AP ========================\n")
        f.write(tabulate(table, headers=headers) + "\n")
        f.write("==========================================================\n")
        f.write('all_mean_aps: ' + str(all_mean_aps) + "\n") 
        f.write('inference time: ' + str(np.mean(inference_time)) + "\n")
    ### NOTE: plot the predictions on the entire dataset ###
    #predictionPlots()    
    





if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()

