from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.base import MemoryModule

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


myfloor = GradFloor.apply

class MyFloor(nn.Module):
    def __init__(self, up=2., t=32):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)

        self.t = t

    def forward(self, x):
        x = x / self.up
        x = myfloor(x * self.t + 0.5) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.up
        return x


class TCL(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Parameter(torch.Tensor([4.]), requires_grad=True)

    def forward(self, x):
        x = F.relu(x, inplace='True')
        x = self.up - x
        x = F.relu(x, inplace='True')
        x = self.up - x
        return x


def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

def replace_activation_by_floor(model, t, threshold):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t, threshold)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                print(module.up.item())
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(module.up.item(), t)
            else:
                if t == 0:
                    model._modules[name] = TCL()
                else:
                    model._modules[name] = MyFloor(threshold, t)
    return model



class ScaledNeuron(MemoryModule):
    def __init__(self, scale=1.):
        super(ScaledNeuron, self).__init__()
        self.scale = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None, backend='torch')

    def forward(self, x):
        x = x / self.scale
        if self.t == 0:
            self.neuron(torch.ones_like(x) * 0.5)
        x = self.neuron(x)
        self.t += 1
        return x * self.scale

    def reset(self):
        self.t = 0
        self.neuron.reset()

def replace_activation_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_neuron(module)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "up"):
                if getattr(module, "unreplaceable") == False:
                    model._modules[name] = ScaledNeuron(scale=module.up.item())
    return model







from spiking_util import yaml_model_load
import math

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def ConvModuleOps(input_mf_size, channel_in, channel_out, kernel_size=1, stride=1., padding=None, dilation=1, group=1): # return ops, output_mf_size
    padding = autopad(kernel_size, padding, dilation)
    if padding is None:
        padding = 0
    return math.ceil((input_mf_size + padding*2 - (kernel_size-1)) / stride) ** 2 * (kernel_size) ** 2 * (channel_in / group) * channel_out, math.ceil((input_mf_size + padding*2 - (kernel_size-1)) / stride)


def BottleNeckModuleOps(input_mf_size, channel_in, channel_out, group=1, shortcut=True, e=0.5, kernel_sizes=(3, 3)):
    c_ = int(channel_out * e)
    add = shortcut and channel_in == channel_out
    conv1_ops, conv1_mf_size = ConvModuleOps(input_mf_size, channel_in, c_, kernel_size=kernel_sizes[0], stride=1)
    conv2_ops, conv2_mf_size = ConvModuleOps(conv1_mf_size, c_, channel_out, kernel_size=kernel_sizes[1], stride=1, group=group)
    return conv1_ops+conv2_ops, conv2_mf_size

def chunk_div(channel_in, chunk=2.):
    chunk_channel = math.ceil(channel_in / chunk)
    chunk_actual_num = math.ceil(channel_in / chunk_channel)
    chunk_channel_mod = chunk_channel - (chunk_actual_num * chunk_channel - channel_in)
    return [chunk_channel for i in range(chunk_actual_num-1)] + [chunk_channel_mod]
def C2fModuleOps(input_mf_size, channel_in, channel_out, bottle_repeat=1, shortcut=False, group=1, e=0.5):
    mid_c = int(channel_out * e)  # hidden channels
    conv1_ops, conv1_mf_size = ConvModuleOps(input_mf_size, channel_in, 2 * mid_c, kernel_size=1, stride=1)
    chunk_channels = chunk_div(2 * mid_c, chunk=2.)
    bottle_ops = 0.
    bottle_fm_size = conv1_mf_size
    for i in range(bottle_repeat):
        bottle_op, bottle_fm_size = BottleNeckModuleOps(conv1_mf_size, mid_c, mid_c, shortcut=shortcut, group=group, kernel_sizes=(3, 3), e=1.0)
        bottle_ops += bottle_op
    conv2_ops, conv2_mf_size = ConvModuleOps(bottle_fm_size, (2 + bottle_repeat) * mid_c, channel_out, kernel_size=1)
    return conv1_ops + bottle_ops + conv2_ops, conv2_mf_size

def SPPFModuleOps(input_mf_size, channel_in, channel_out, kernel_size=5):
    mid_c = channel_in // 2
    conv1_ops, conv1_mf_size = ConvModuleOps(input_mf_size, channel_in, mid_c, kernel_size=1, stride=1)
    #avg pool
    padding = kernel_size // 2
    avgpool_mf_size_1 =  (conv1_mf_size + 2 * padding - kernel_size) // 1 + 1
    avgpool_mf_size_2 =  (avgpool_mf_size_1 + 2 * padding - kernel_size) // 1 + 1

    conv2_ops, conv2_mf_size = ConvModuleOps(avgpool_mf_size_2, mid_c * 4, channel_out, kernel_size=1, stride=1)
    return conv1_ops + conv2_ops, conv2_mf_size

def upsample(input_mf_size, factor=2):
    return input_mf_size * factor

def dfl(input_mf_size, channel_in=16):
    return input_mf_size**2, input_mf_size

def DectModuleOps(input_mf_size=(), nc=9, ch=()):
    reg_max = 16
    nl = len(ch)
    no = nc + reg_max * 4
    c2, c3 = max((16, ch[0] // 4, reg_max * 4)), max(ch[0], min(nc, 100))  # channels
    ops_cv2 = 0
    fm_cv2 = []
    for x, fm_size in zip(ch, input_mf_size):
        ops_cv2_Conv1, cv2_Conv1_fm_size = ConvModuleOps(fm_size, x, c2, 3)
        ops_cv2_Conv2, cv2_Conv2_fm_size = ConvModuleOps(cv2_Conv1_fm_size, c2, c2, 3)
        ops_cv2_Conv3, cv2_Conv3_fm_size = ConvModuleOps(cv2_Conv2_fm_size, c2, 4 *  reg_max, 1)
        ops_cv2 += (ops_cv2_Conv1 + ops_cv2_Conv2 + ops_cv2_Conv3)
        fm_cv2.append(cv2_Conv3_fm_size)

    ops_cv3 = 0
    fm_cv3 = []
    for x, fm_size in zip(ch, input_mf_size):
        ops_cv3_Conv1, cv3_Conv1_fm_size = ConvModuleOps(fm_size, x, c3, 3)
        ops_cv3_Conv2, cv3_Conv2_fm_size = ConvModuleOps(cv3_Conv1_fm_size, c3, c3, 3)
        ops_cv3_Conv3, cv3_Conv3_fm_size = ConvModuleOps(cv3_Conv2_fm_size, c3, nc, 1)
        ops_cv3 += (ops_cv3_Conv1 + ops_cv3_Conv2 + ops_cv3_Conv3)
        fm_cv3.append(cv3_Conv3_fm_size)

    return ops_cv3 + ops_cv2









ac_op_energy = 0.9     # 10^-9 mJ
mac_op_energy = 4.6     #mJ 10^-9 mJ
avg_firing_rate = 1/6 * 8
def calculateOps():
    input_mf_size = 1024
    d=1
    w=1
    r=1
    layer_ops = []



    layer_12_ops, _ = C2fModuleOps(input_mf_size//12, round(512 * w * (1+r)), round(521 * w), bottle_repeat=round(3*d))
    layer_15_ops, _ = C2fModuleOps(input_mf_size // 8, round(768 * w), round(256 * w), bottle_repeat=round(3*d))
    layer_16_ops, _ = ConvModuleOps(input_mf_size // 8, round(256 * w), round(256 * w), kernel_size=3, stride=2.)
    layer_18_ops, _ = C2fModuleOps(input_mf_size // 16, round(768 * w), round(512 * w), bottle_repeat=round(3*d))
    layer_19_ops, _ = ConvModuleOps(input_mf_size // 16, round(512 * w), round(512 * w), kernel_size=3, stride=2.)
    layer_21_ops, _ = C2fModuleOps(input_mf_size // 32, round(512 * w * (1+r)), round(512 * w * r), bottle_repeat=round(3*d))

    head_ops = DectModuleOps((input_mf_size // 8, input_mf_size // 16, input_mf_size // 32), nc=7, ch=(round(256 * w), round(512 * w), round(512 * w * r)))


    replaced_ops = layer_12_ops + layer_15_ops + layer_16_ops + layer_18_ops + layer_19_ops + layer_21_ops + head_ops
    snn_ops = (layer_12_ops + layer_15_ops + layer_16_ops + layer_18_ops + layer_19_ops + layer_21_ops + head_ops) * avg_firing_rate
    print(replaced_ops / 1000000000, 'GFlops')
    print(snn_ops / 1000000000, 'GFlops')
    print(257.6 * mac_op_energy, 'ANN XL mJ')
    print(164.8 * mac_op_energy, 'ANN L mJ')
    print( 47.6 * mac_op_energy +  ac_op_energy * (snn_ops / 1000000000), 'SNN mJ')


def main():
    model = YOLO('yolov8n-pose.yaml').load('runs/pose/train6/weights/best.pt')
    # snn conversion
    print('converting cnn to snn')
    snn = replace_activation_by_neuron(model)
    snn.model.spiking_mode = True
    print(snn)
    # torch.save(snn, 'weights/snn.pt')
    print('validate yolo v8 pose')
    metrics = model.val(data='coco-pose.yaml', batch=128)
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75


    # calculateOps('yolov8l.yaml')

def train():
    model = YOLO('yolov8n.pt')
    results = model.train(data='coco.yaml', batch=128, epochs=100, imgsz=640, device='0,1,2,3')
    metrics = model.val(data='coco.yaml')
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75

def train_pose():
    model = YOLO("yolov8n-pose.pt")
    results = model.train(data="coco-pose.yaml",batch=256, epochs=100, imgsz=640, device='0,1,2,3')
    metrics = model.val(data='coco-pose.yaml')
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75

def predicter():
    # model = YOLO('yolov8n-pose.yaml').load('runs/pose/train6/weights/best.pt')
    model = YOLO('runs/pose/train14/weights/best.pt')
    model.spiking_mode = True
    snn = replace_activation_by_neuron(model)
    snn.model.spiking_mode = True
    
    # print(f"snn is {snn}")
    results = model.predict(source="data/img0001.png", save=True)
    
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs, boxes are torch.Size([300, 6])
    #     keypoints = result.keypoints  # Keypoints object for pose outputs, keypoints are torch.Size([300, 17, 3])
        # print(f"boxes are {boxes}\nkeypoints are {keypoints}\n")
        # result.save(filename='result.jpg')  # save to disk

def val_ann():
    model = YOLO('runs/pose/train14/weights/best.pt')
    model.spiking_mode = True
    snn = replace_activation_by_neuron(model)
    snn.model.spiking_mode = True
    print('validate yolo v8 pose')
    metrics = model.val(data='coco-pose.yaml', batch=1)
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    # main()
    # train_pose()
    # calculateOps()
    predicter()
    # train()
    # val_ann()