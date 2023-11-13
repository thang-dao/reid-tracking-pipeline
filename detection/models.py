import os
import cv2
import time
import torch
import numpy as np
import pandas as pd

from ultralytics.yolo.data.augment import LetterBox
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.yolo.utils.checks import check_imgsz
from yolov8.ultralytics.yolo.utils.torch_utils import select_device
from yolov8.ultralytics.yolo.utils.ops import non_max_suppression, scale_boxes


class YoloV8:
    def __init__(self, device, yolo_weights, imgsz=(640,640), conf_thres=0.2, iou_thres=0.5):
        self.dnn = False
        self.half = False
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = None
        self.agnostic_nms=True
        self.max_det = 1000
        self.imgsz = imgsz
        self.device = select_device(device)
        self.model = AutoBackend(yolo_weights, device=self.device, dnn=self.dnn, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_imgsz(self.imgsz, stride=self.stride)  # check image size
        self.letter_box = LetterBox(self.imgsz, self.pt, stride=self.stride)
    
    def _preprocess(self, img):
        # im = LetterBox(self.imgsz, self.pt, stride=self.stride)(image=img)
        im = self.letter_box(image=img)
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        return im

    def _img2tensor(self, im, device, half=False):
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im
    
    def predict(self, imgs):
        ims = [self._preprocess(im) for im in imgs]
        tensor = self._img2tensor(np.array(ims), self.device)
        preds = self.model(tensor, augment=False, visualize=False)
        p = non_max_suppression(preds, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dets = []
        for i,det in enumerate(p):
            if det is not None and len(det):
                det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], imgs[i].shape).round()
            det = det.cpu().numpy()
            dets.append(det)
            # draw_trackers(det, imgs[i])
        return dets
        

def create_detection_model(model_name, cfg, device):
    if model_name == 'YoloV8':
        model = YoloV8(device=device, yolo_weights=cfg.get('WEIGHTS'), imgsz=cfg.get('IMGSZ'), conf_thres=cfg.get('CONF_THRESHOLD'), iou_thres=cfg.get('IOU_THRESHOLD'))
    return model