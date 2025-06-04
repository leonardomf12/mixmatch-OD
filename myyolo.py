# (C) 2025 Ricardo Cruz <rpcruz@fe.up.pt>
# YOLO extracted
# credits to Celso Pereira https://github.com/CelsoPereira1
# also see https://github.com/ultralytics/ultralytics/issues/19991

import math
import torch.nn as nn
import ultralytics
from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.utils.ops import non_max_suppression

class MyYOLO(nn.Module):
    def __init__(self, name, nc=80, max_det=300, input_resolution=640):
        super().__init__()
        reg_max = 16 # for each bounding box regression the model will predict a probability distribution over N possible values for each coordinate, default: 16
        no = nc + reg_max * 4
        self.yolo = ultralytics.YOLO(name) # load the model - pretrained *.pt models as well as configuration *.yaml files can be passed

        # adapt model to the number of classes in the dataset
        self.yolo.model.model[-1].nc = nc
        self.yolo.model.model[-1].max_det = max_det
        self.yolo.model.model[-1].no = no
        self.yolo.model.model[-1].reg_max = reg_max
        
        # head stride: aligns the scale of predicted bounding boxes and anchor points with the ground-truth boxes and labels
        # self.yolo.model.model[-1].stride = torch.tensor([8., 16., 32.]) # default: tensor([ 8., 16., 32.]) # do NOT change
        self.ch = [self.yolo.model.model[:-1][i].cv2.conv.out_channels for i in self.yolo.model.model[-1].f]
        c2, c3 = max((16, self.ch[0] // 4, reg_max * 4)), max(self.ch[0], min(nc, 100)) # channels

        self.yolo.model.model[-1].cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in self.ch)
        self.yolo.model.model[-1].cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in self.ch) # this differs from the YOLOv10, plus, as the YOLOv10 is an end2end model, YOLOv10 has also one2one_cv2 and one2one_cv3
        self.yolo.model.model[-1].dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        # note: in this Ultralytics function bias_init(), the resolution 640 is hardcoded as a reference resolution. If a different image resolution is used, 
        # the fixed value of 640 will no longer match the actual input resolution. This mismatch could make the initial bias suboptimal
        # init weights, biases
        for a, b, s in zip(self.yolo.model.model[-1].cv2, self.yolo.model.model[-1].cv3, self.yolo.model.model[-1].stride):
            a[-1].bias.data[:] = 1.0 # default 1.0
            b[-1].bias.data[: self.yolo.model.model[-1].nc] = math.log(5 / self.yolo.model.model[-1].nc / (input_resolution / s) ** 2) # default math.log(5 / self.original_model.model.model[-1].nc / (640 / s) ** 2)
        initialize_weights(self.yolo.model.model[-1])

    def forward(self, x, predict=False, conf_thres=0.25, iou_thres=0.45):
        # see https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py
        y = []

        # YOLO backbone forward pass
        for b in self.yolo.model.model[:-1]: # m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            if b.f != -1:  # if not from previous layer
                x = y[b.f] if isinstance(b.f, int) else [x if j == -1 else y[j] for j in b.f] # from earlier layers
            x = b(x) # run
            y.append(x if b.i in self.yolo.model.save else None) # save output

        # YOLO head forward pass
        embed_single = [y[j] for j in self.yolo.model.model[-1].f]
        x = self.yolo.model.model[-1](embed_single)

        # this 'x' is different, depending on the mode:
        # - train: multi-scale list with cell predictions [(B, C, H, W)]
        # - eval: tuple with:
        #    - x[0]: list len=batch_size where each is tensor with shape (num_classes+4,H*W)
        #    - x[1]: same as train

        if predict:
            # post-process
            x = non_max_suppression(x[0], conf_thres=conf_thres, iou_thres=iou_thres)
            # NMS returns list len=batch_size with (num_objects, num_classes+1+1)
            # the last two 1+1 are class scores and class number
            return [{'labels': y[:, -1].long(), 'scores': y[:, -2], 'boxes': y[:, :4]} for y in x]
        return x

    def set_trainable(self):
        self.yolo.model.model[:-1].train()
        self.yolo.model.model[-1].train()

    def set_eval(self):
        self.yolo.model.model[:-1].eval()
        self.yolo.model.model[-1].eval()