
import time

import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

import utils.radar_loader as loader
import utils.drawer as drawer
import glob, os
import shutil
from tqdm import tqdm
from Load_radar_data import getitem_RADframe,loadDataForPlot,RADData_parse_for_one_frame
from utils.utils import resize_radar_data

if __name__ == "__main__":
    yolo = YOLO()

    mode = "radar_predict"

    crop            = False
    count           = False

    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    test_interval   = 100
    fps_image_path  = "img/street.jpg"

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    heatmap_save_path = "model_data/heatmap_vision.png"

    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    
    target_shape    = [256,256,256]
    interpolation   = 'nearest'
    
    
    
    
    
    if mode == "radar_predict":

                
        ### NOTE: inference starting from here ###
        config = loader.readConfig()
        config_data  = config["DATA"]
        config_radar = config["RADAR_CONFIGURATION"]
        config_model = config["MODEL"]
        config_inference = config["INFERENCE"]
        
        #---------------------------#
        #   Split train and validation
        #---------------------------#
        RAD_sequences_train = loader.readSequences(config_data,mode="train")
        RAD_sequences_test = loader.readSequences(config_data,mode="test")
        RAD_sequences_train, RAD_sequences_validate = loader.splitTrain(RAD_sequences_train,validate=True)
        num_train   = len(RAD_sequences_train)
        num_val     = len(RAD_sequences_validate) 
        num_test    = len(RAD_sequences_test)    
        
        #all_RAD_files = glob.glob(os.path.join(config_data["test_set_dir"], "RAD/*/*.npy"))
        dataset_dir="train_set_dir"
        #all_RAD_files = glob.glob(os.path.join(config_data[dataset_dir], "RAD/*/*.npy"))
        all_RAD_files = RAD_sequences_train
        
        image_save_dir = "./lei_outputs/inference_plots/"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        else:
            shutil.rmtree(image_save_dir)
            os.makedirs(image_save_dir)
        print("Start plotting, it might take a while...")
        pbar = tqdm(total=len(all_RAD_files))
        model_RAD_st = []
        model_cart_st = []

        frame_idx = 0
        RAD_frame, RAD_label_frame = getitem_RADframe(all_RAD_files,frame_idx,config_data, 
                                                      yolo.input_shape, target_shape, interpolation,
                                                      mode = 'RA', dataset_dir="train_set_dir")
        data = np.expand_dims(RAD_frame, 0) # add batchsize
        nms_pred, r_image = yolo.detect_image(data, crop = crop, count = count)
        # #r_image.show()
        
        for sequence_num, data, stereo_left_image, RD_img, RA_img, RA_cart_img in \
                loadDataForPlot(all_RAD_files, config_data, config_inference, config_radar,dataset_dir):
             
            if data is None or stereo_left_image is None:
                pbar.update(1)
                continue

            model_RAD_time_start = time.time()
            
            nms_pred,r_image = yolo.detect_image(data, crop = crop, count = count)
            RAD_frame, RAD_label_frame = RADData_parse_for_one_frame(all_RAD_files,sequence_num,config_data,dataset_dir="train_set_dir")
            RAD_label_frame = np.concatenate( (RAD_label_frame[RAD_label_frame[...,0]>0],RAD_label_frame[RAD_label_frame[...,0]>0][:,-1:]), axis=-1)
            #r_image.show()
            

            model_RAD_st.append(time.time() - model_RAD_time_start)
            
            ### Plot
            fig, axes = drawer.prepareFigure(3, figsize=(80, 6))
            colors = loader.randomColors(config_data["all_classes"])
            nms_pred_cart = None
            drawer.clearAxes(axes)
            ## draw bbox
            drawer.drawInference(stereo_left_image, RD_img, \
                    RA_img, RA_cart_img, nms_pred[0], \
                    config_data["all_classes"], colors, axes, \
                    radar_cart_nms=nms_pred_cart)
                
            drawer.saveFigure(image_save_dir, "%.6d.png"%(sequence_num))

            pbar.update(1)
        print("------", " The average inference time for RAD Boxes: ", \
                                                np.mean(model_RAD_st))
        print("======", " The average inference time for Cartesian Boxes: ", \
                                                np.mean(model_cart_st))
                
    

    if mode == "predict":

        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Error")

        fps = 0.0
        while(True):
            t1 = time.time()

            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(yolo.detect_image(frame))

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
                
    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
