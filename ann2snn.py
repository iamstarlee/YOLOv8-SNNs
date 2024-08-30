import torch
import cv2
from ultralytics import YOLO
from spikingjelly.activation_based import ann2snn


if __name__ == '__main__':
    # pose_model = torch.load('model_data/yolov8s-pose.pt')
    model = YOLO("yolov8s-pose.pt")
    model_converter = ann2snn.Converter(mode='max', dataloader=train_data_loader)
    snn_model = model_converter(model)
    imgs = cv2.imread('imgs_in/img0001.png')
    result = model(imgs, save = True)
