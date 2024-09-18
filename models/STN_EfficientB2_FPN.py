

""" 

FPN network Classes

Author: Gurkirt Singh
Inspired from https://github.com/kuangliu/pytorch-retinanet and
https://github.com/gurkirt/realtime-action-detection

"""
from modules.anchor_box_retinanet import anchorBox
from modules.detection_loss import MultiBoxLoss, YOLOLoss, FocalLoss
from modules.box_utils import decode
import torch, math, pdb, math
import torch.nn as nn



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models

layers = [6, 12, 48, 32]  
use_bias = False
seq_len = 16
modelname = 'efficientnetb2'  

class CustomDenseNet201(nn.Module):
    def __init__(self, seq_len):
        super(CustomDenseNet201, self).__init__()
        densenet201 = torchvision_models.efficientnet_b2(pretrained=True)
        self.features = densenet201.features
        self.seq_len = seq_len

        # Define Spatial Transformer Network (STN) module
        self.stn = SpatialTransformerModule()

    def forward(self, x):
        # Apply STN to input images
        x_transformed = self.stn(x)

        # Pass transformed images through the backbone network
        features = self.features(x_transformed)

        return features

class DenseNetFPN(nn.Module):
    def __init__(self, densenet, seq_len):
        super(DenseNetFPN, self).__init__()
        self.seq_len = seq_len
        self.features = densenet.features
        self.conv6 = nn.Conv2d(1408, 720, kernel_size=3, stride=2, padding=1, bias=use_bias)  # P6
        self.conv7 = nn.Conv2d(720, 720, kernel_size=3, stride=2, padding=1, bias=use_bias)  # P7

        self.lateral_layer1 = nn.Conv2d(1408, 720, kernel_size=1, stride=1, bias=use_bias)
        self.lateral_layer2 = nn.Conv2d(1408, 720, kernel_size=1, stride=1, bias=use_bias)
        self.lateral_layer3 = nn.Conv2d(1408, 720, kernel_size=1, stride=1, bias=use_bias)

        self.corr_layer1 = nn.Conv2d(720, 720, kernel_size=3, stride=1, padding=1, bias=use_bias)  # P4
        self.corr_layer2 = nn.Conv2d(720, 720, kernel_size=3, stride=1, padding=1, bias=use_bias)  # P4
        self.corr_layer3 = nn.Conv2d(720, 720, kernel_size=3, stride=1, padding=1, bias=use_bias)  # P3

        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(720)
        self.dropout = nn.Dropout(0.5)

    def _upsample(self, x, y):
        _, _, h, w = y.size()
        x_upsampled = F.interpolate(x, size=(h, w), mode='nearest')
        return x_upsampled

    def forward(self, x):
        x = self.features(x)
        c1 = x
        c2 = x
        c3 = x
        c4 = x

        # Top-down
        p5 = self.lateral_layer1(c4)
        p5_upsampled = self._upsample(p5, c3)
        p5 = self.corr_layer1(p5)

        p4 = self.lateral_layer2(c3)
        p4 = p5_upsampled + p4
        p4_upsampled = self._upsample(p4, c2)
        p4 = self.corr_layer2(p4)

        p3 = self.lateral_layer3(c2)
        p3 = p4_upsampled + p3
        p3 = self.corr_layer3(p3)

        p6 = self.conv6(c4)
        p6 = self.relu(p6)
        p7 = self.conv7(p6)
        p7 = self.relu(p7)

        # Batch normalization and dropout
        p7 = self.batch_norm(p7)
        p7 = self.dropout(p7)

        # Upsample P3 and P4 to higher scales
        p3_upsampled = F.interpolate(p3, scale_factor=4, mode='nearest')
        p4_upsampled = F.interpolate(p4, scale_factor=2, mode='nearest')

        return p3_upsampled, p4_upsampled, p5, p6, p7

class SpatialTransformerModule(nn.Module):
    def __init__(self):
        super(SpatialTransformerModule, self).__init__()
        # Define the spatial transformer network layers
        self.localization_network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 2 * 3)  # 2x3 affine matrix
        )
        # Initialize the weights/biases of the fc_loc layers
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Compute the transformation matrix theta
        xs = self.localization_network(x)
        xs = xs.view(-1, 128 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Apply the spatial transformation to the input
        grid = F.affine_grid(theta, x.size())
        x_transformed = F.grid_sample(x, grid)

        return x_transformed

def densenet_fpn(name, use_bias, seq_len=1):
    if name == 'efficientnetb2':
        densenet = CustomDenseNet201(seq_len)
        return DenseNetFPN(densenet, seq_len)
    else:
        raise ValueError("Unsupported DenseNet variant")


class RetinaNet(nn.Module):
    """Feature Pyramid Network Architecture
    The network is composed of a backbone FPN network followed by the
    added Head conv layers.  
    Each head layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions

    See: 
    RetinaNet: https://arxiv.org/pdf/1708.02002.pdf for more details.
    FPN: https://arxiv.org/pdf/1612.03144.pdf

    Args:
        backbone Network:
        Program Argument Namespace

    """

    def __init__(self, backbone, args):
        super(RetinaNet, self).__init__()

        self.num_classes = args.num_classes
        # TODO: implement __call__ in 
        
        self.anchors = anchorBox()
        self.ar = self.anchors.ar
        args.ar = self.ar
        self.use_bias = args.use_bias
        self.head_size = args.head_size
        self.backbone_net = backbone
        self.shared_heads = args.shared_heads
        self.num_head_layers = args.num_head_layers
        
        assert self.shared_heads<self.num_head_layers, 'number of head layers should be less than shared layers h:'+str(self.num_head_layers)+' sh:'+str(self.shared_heads)
        
        if self.shared_heads>0:
            self.features_layers = self.make_features(self.shared_heads)
        self.reg_heads = self.make_head(self.ar * 4, self.num_head_layers - self.shared_heads)
        self.cls_heads = self.make_head(self.ar * self.num_classes, self.num_head_layers - self.shared_heads)
        
        if args.loss_type != 'mbox':
            self.prior_prob = 0.01
            bias_value = -math.log((1 - self.prior_prob ) / self.prior_prob )
            nn.init.constant_(self.cls_heads[-1].bias, bias_value)
        if not hasattr(args, 'eval_iters'): # eval_iters only in test case
            if args.loss_type == 'mbox':
                self.criterion = MultiBoxLoss(args.positive_threshold)
            elif args.loss_type == 'yolo':
                self.criterion = YOLOLoss(args.positive_threshold, args.negative_threshold)
            elif args.loss_type == 'focal':
                self.criterion = FocalLoss(args.positive_threshold, args.negative_threshold)
            else:
                error('Define correct loss type')


    def forward(self, images, gts=None, counts=None,get_features=False):
        sources = self.backbone_net(images)
        features = list()
        # pdb.set_trace()
        if self.shared_heads>0:
            for x in sources:
                features.append(self.features_layers(x))
        else:
            features = sources
        
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        ancohor_boxes = self.anchors(grid_sizes)
        
        loc = list()
        conf = list()
        
        for x in features:
            loc.append(self.reg_heads(x).permute(0, 2, 3, 1).contiguous())
            conf.append(self.cls_heads(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        
        flat_loc = loc.view(loc.size(0), -1, 4)
        flat_conf = conf.view(conf.size(0), -1, self.num_classes)
        # pdb.set_trace()
        if get_features: # testing mode with feature return
            return  torch.stack([decode(flat_loc[b], ancohor_boxes) for b in range(flat_loc.shape[0])], 0), flat_conf, features
        elif gts is not None: # training mode 
            return self.criterion(flat_conf, flat_loc, gts, counts, ancohor_boxes)
        else: # otherwise testing mode 
            return  torch.stack([decode(flat_loc[b], ancohor_boxes) for b in range(flat_loc.shape[0])], 0), flat_conf


    def make_features(self,  shared_heads):
        layers = []
        use_bias =  self.use_bias
        head_size = self.head_size
        for _ in range(shared_heads):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1, padding=1, bias=use_bias))
            layers.append(nn.ReLU(True))
        
        layers = nn.Sequential(*layers)
        
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

    def make_head(self, out_planes, nun_shared_heads):
        layers = []
        use_bias =  self.use_bias
        head_size = self.head_size
        for _ in range(nun_shared_heads):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1, padding=1, bias=use_bias))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(head_size, out_planes, kernel_size=3, stride=1, padding=1))
        layers = nn.Sequential(*layers)
        
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

def build_retinanet_shared_heads(args):
    # print('basenet', args.basenet)
    backbone = densenet_fpn(modelname, use_bias)
    # print('backbone model::', backbone)
    return RetinaNet(backbone, args)
