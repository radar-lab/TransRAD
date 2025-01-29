import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox
from utils.helper import NMS_3D_Lei, NMS_2D_Lei, NMS_3D_Lei_overlap

class YOLO(object):
    _defaults = {

        "model_path"        : 'model_data/best_epoch_weights.pth',
        "classes_path"      : 'model_data/radar_classes.txt',

        "input_shape"       : [256, 256, 64],

        "phi"               : 'n',

        "confidence"        : 0.3, #0.4, #0.2, #0.001, #0.5,

        "nms_iou"           : 0.1, #0.3,  #0.1 for 3D; 0.3 for 2D

        "letterbox_image"   : False,

        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 

        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.bbox_util                      = DecodeBox(self.num_classes, self.input_shape)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)


    def generate(self, onnx=False):

        self.net    = YoloBody(self.input_shape, self.num_classes, self.phi)
        
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()


    def detect_image(self, image, mode='RA', crop = False, count = False, plot = False):

        if len(image.shape) == 3:
            image  = np.expand_dims(image, 0)

            image_shape = np.array(np.shape(image)[0:-1])
        else:

            image_shape = np.array(np.shape(image)[1:-1]) 
            
        with torch.no_grad():            
            images = torch.from_numpy(np.array(image)).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images) #outputs = dbox, cls, x, anchors, strides
            
            #####################   Lei-test    ########################
            outputs = self.bbox_util.decode_box_Lei(outputs, mode)
            outputs = outputs.cpu().detach().numpy()
            #---------------------------------------------------------#
            #  best_bboxes = xyzwhdsc -> [num_bbox, box + confidence_score + class]
            #---------------------------------------------------------#
            #best_bboxes = NMS_2D_Lei(outputs[0], self.input_shape, self.nms_iou, self.confidence, sigma=0.3, method='nms')
            best_bboxes = NMS_3D_Lei(outputs[0], self.input_shape, self.nms_iou, self.confidence, sigma=0.3, method='nms')
            best_bboxes = NMS_3D_Lei_overlap(best_bboxes, self.input_shape, iou_threshold=0.1)
            out_boxes, out_scores, out_classes = best_bboxes[...,:6],best_bboxes[...,6:7],best_bboxes[...,7:8] 
                                       
            ######################################################################
            top_label = out_classes[:, 0]
            top_conf = out_scores[:, 0]
            top_boxes = out_boxes

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.shape[2] + 0.5).astype('int32'))
        thickness   = int(max((image.shape[1] + image.shape[2]) // np.mean(self.input_shape), 1))

        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        if plot:
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box             = top_boxes[i]
                score           = top_conf[i]
    
                top, left, bottom, right = box
    
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.shape[2], np.floor(bottom).astype('int32'))
                right   = min(image.shape[1], np.floor(right).astype('int32'))
    
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
    
                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

        return best_bboxes,image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():

                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)

                results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            dbox, cls, x, anchors, strides = self.net(images)
            outputs = [xi.split((xi.size()[1] - self.num_classes, self.num_classes), 1)[1] for xi in x]
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, -1, h, w]), [0, 2, 3, 1])[0]
            score      = np.max(sigmoid(sub_output[..., :]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
