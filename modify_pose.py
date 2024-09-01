import torch

if __name__ == '__main__':
    model = torch.load('model_data/yolov8s-pose.pt')['model']
    print(f"model is {model}")