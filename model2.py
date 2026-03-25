import torch
import torch.nn as nn
import torch.nn.functional as F

class TopDownFusionBlock(nn.Module):
    def __init__(self, channels_low, channels_high, channels_out):
        super(TopDownFusionBlock, self).__init__()
        
        self.reduce_low = nn.Conv2d(channels_low, channels_out, kernel_size=1, bias=False)
        
        if channels_high != channels_out:
             self.reduce_high = nn.Conv2d(channels_high, channels_out, kernel_size=1, bias=False)
        else:
             self.reduce_high = nn.Identity()
        
        self.post_add = nn.Sequential(
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, groups=channels_out, bias=False), 
            nn.BatchNorm2d(channels_out), 
            nn.Conv2d(channels_out, channels_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_out) 
        )

    def forward(self, c_low, c_high):
        
        c_high_up = F.interpolate(c_high, size=(c_low.size(2), c_low.size(3)), mode='nearest')
        c_high_matched = self.reduce_high(c_high_up)
        
        c_low_reduced = self.reduce_low(c_low)
        
        merged = c_low_reduced + c_high_matched
        
        return self.post_add(merged)

class ModifiedMobileNet(nn.Module):
    def __init__(self):
        super(ModifiedMobileNet, self).__init__()
        
        self.features_to_conv5 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=4, padding=1) 
        )
        
        self.conv_6_to_8 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        
        self.conv_9_to_11 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_12_to_13 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features_to_conv5(x)
        c8 = self.conv_6_to_8(x)     
        c11 = self.conv_9_to_11(c8)  
        c13 = self.conv_12_to_13(c11)
        return c8, c11, c13

class HandDetectionNetwork(nn.Module):
    def __init__(self, num_anchors=6):
        super(HandDetectionNetwork, self).__init__()
        self.backbone = ModifiedMobileNet()
        
        self.extra_conv_14 = nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1) 
        self.extra_conv_15 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)  
        self.extra_conv_16 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)  
        
        self.td_fusion_15_to_14 = TopDownFusionBlock(512, 256, 512)
        self.td_fusion_14_to_13 = TopDownFusionBlock(1024, 512, 1024)
        self.td_fusion_13_to_11 = TopDownFusionBlock(512, 1024, 512)
        self.td_fusion_11_to_8  = TopDownFusionBlock(256, 512, 256)
        
        self.num_anchors = num_anchors
        
        def build_heads(in_channels):
            return nn.ModuleDict({
                'cls': nn.Conv2d(in_channels, num_anchors, kernel_size=3, padding=1), 
                'loc': nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)  
            })

        self.heads_8 = build_heads(256)
        self.heads_11 = build_heads(512)
        self.heads_13 = build_heads(1024)
        self.heads_14 = build_heads(512)
        self.heads_15 = build_heads(256)
        self.heads_16 = build_heads(128)

    def forward(self, x):
        
        c8, c11, c13 = self.backbone(x)
        
        c14 = self.extra_conv_14(c13)
        c15 = self.extra_conv_15(c14)
        c16 = self.extra_conv_16(c15)
        
        c14_star = self.td_fusion_15_to_14(c14, c15)
        c13_star = self.td_fusion_14_to_13(c13, c14_star)
        c11_star = self.td_fusion_13_to_11(c11, c13_star)
        c8_star  = self.td_fusion_11_to_8(c8, c11_star)
        
        features = [c8_star, c11_star, c13_star, c14_star, c15, c16]
        heads_list = [self.heads_8, self.heads_11, self.heads_13, self.heads_14, self.heads_15, self.heads_16]
        
        predictions = {'cls': [], 'loc': []}
        
        for feature, head in zip(features, heads_list):
            predictions['cls'].append(head['cls'](feature))
            predictions['loc'].append(head['loc'](feature))
            
        return predictions
  
from torchvision.ops import box_iou

def cxcy_to_xy(cxcy):
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)

def encode_boxes(matched_truths, priors):
    variance = [0.1, 0.2] 
    
    g_cxcy = (matched_truths[:, :2] + matched_truths[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variance[0] * priors[:, 2:])
    
    eps = 1e-5
    g_wh = (matched_truths[:, 2:] - matched_truths[:, :2]) / torch.clamp(priors[:, 2:], min=eps)
    g_wh = torch.log(torch.clamp(g_wh, min=eps)) / variance[1]
    
    return torch.cat([g_cxcy, g_wh], 1)  
class HandTrackerLoss(nn.Module):
    def __init__(self, anchors_cxcy, threshold=0.5, weight_box=5.0):
        super(HandTrackerLoss, self).__init__()
        self.threshold = threshold
        self.weight_box = weight_box
        
        self.priors_cxcy = anchors_cxcy
        self.priors_xy = cxcy_to_xy(anchors_cxcy)

    def forward(self, out, labels):
        preds_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1) 
                               for o in out["cls"]], dim=1) 
                               
        preds_loc = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1, 4) 
                               for o in out["loc"]], dim=1) 

        batch_size = preds_cls.size(0)
        num_priors = self.priors_cxcy.size(0)
        
        
        target_conf = torch.zeros(batch_size, num_priors, device=preds_cls.device)
        target_loc = torch.zeros(batch_size, num_priors, 4, device=preds_loc.device)

        for idx in range(batch_size):
            truths = labels[idx][:, 1:] 
            
            if truths.size(0) == 0:
                continue
                
            overlaps = box_iou(truths, self.priors_xy.to(truths.device))
            
            best_truth_overlap, best_truth_idx = overlaps.max(0)
            
            matched_truths = truths[best_truth_idx]
            
            target_loc[idx] = encode_boxes(matched_truths, self.priors_cxcy.to(truths.device))
            
            target_conf[idx][best_truth_overlap > self.threshold] = 1.0

        greutate_pozitiva = torch.tensor([75.0], device=preds_cls.device)
        
        loss_conf = F.binary_cross_entropy_with_logits(preds_cls, target_conf, pos_weight=greutate_pozitiva)
        
        loss_box_raw = F.smooth_l1_loss(preds_loc, target_loc, reduction='none').sum(dim=-1)
        
        
        pos_mask = target_conf > 0
        loss_box_masked = (loss_box_raw * pos_mask).sum() / (pos_mask.sum() + 1e-5)

        return loss_conf + (self.weight_box * loss_box_masked)
    
    
    