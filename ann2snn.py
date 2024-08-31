import torch
import cv2
from ultralytics import YOLO
from spikingjelly.activation_based import ann2snn
from nets.backbone import Backbone
import torchvision.datasets as dset
import torchvision.transforms as transforms

if __name__ == '__main__':
    # pose_model = torch.load('model_data/yolov8s-pose.pt')
    # model = Backbone(base_channels=32, base_depth=1, phi='s', deep_mul=1.0, pretrained=False)
    # model.load_state_dict(torch.load('model_data/yolov8_s_backbone_weights.pt'))
    # imgs = cv2.imread('imgs_in/img0001.png')
    # print(model)
    dataset_train = dset.CIFAR10(root="datasets", train=True, download=True, transform=transforms.ToTensor())
    train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size = 64)
