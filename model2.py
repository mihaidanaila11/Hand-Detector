import torch
import torch.nn as nn
import torch.nn.functional as F

class TopDownFusionBlock(nn.Module):
    def __init__(self, channels_low, channels_high_star):
        super(TopDownFusionBlock, self).__init__()
        
        # 1. Reducem adâncimea canalelor pentru C_low (Fig 2: 1x1 Conv) [cite: 146]
        # channels_high_star reprezintă canalele hărții superioare deja procesate
        self.reduce_low = nn.Sequential(
            nn.Conv2d(channels_low, channels_high_star, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_high_star) # Adăugat pentru consistență 
        )
        
        # 2. Rafinarea după adunare (Fig 2: cascada de convoluții) 
        self.post_add = nn.Sequential(
            # Depthwise 3x3 [cite: 150, 157]
            nn.Conv2d(channels_high_star, channels_high_star, kernel_size=3, 
                      padding=1, groups=channels_high_star, bias=False),
            nn.BatchNorm2d(channels_high_star),
            nn.ReLU(inplace=True),
            
            # Pointwise 1x1 [cite: 150, 156]
            nn.Conv2d(channels_high_star, channels_high_star, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_high_star),
            nn.ReLU(inplace=True)
        )

    def forward(self, c_low, c_high_star):
        # A. Upsampling (2x) pentru harta superioară (Chigh*) [cite: 145, 162]
        up_high = F.interpolate(c_high_star, size=(c_low.size(2), c_low.size(3)), mode='nearest')
        
        # B. Procesare C_low [cite: 146]
        low_feat = self.reduce_low(c_low)
        
        # C. Element-wise addition [cite: 147, 161]
        merged = low_feat + up_high
        
        # D. Generarea noii hărți C_low* [cite: 147, 159]
        return self.post_add(merged)

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.convBlock(x)
        
        


class ModifiedMobileNet(nn.Module):
    def __init__(self):
        super(ModifiedMobileNet, self).__init__()
        
        # Input: 300x300x3
        # Output: 150x150x32
        
        self.img_6 = nn.Sequential(
            # conv 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
   
                
            # Input: 150x150x32 -> Output: 150x150x64 conv2 
            ConvBlock(32, 64, stride=1),
            
            # Input: 150x150x64 -> Output: 75x75x128 conv3
            ConvBlock(64, 128, stride=2),
            
            # Input: 75x75x128 -> Output: 75x75x128 conv 4
            ConvBlock(128, 128, stride=1),
            
            # Input: 75x75x128 -> Output: 38x38x256 conv 5
            ConvBlock(128, 256, stride=2),
            
            # Input: 38x38x256 -> Output: 38x38x512 (MĂRIME PĂSTRATĂ) conv 6
            ConvBlock(256, 512, stride=1)
        
        )
        
        # Input: 38x38x512 -> Output: 38x38x512 (MĂRIME PĂSTRATĂ) conv 7
        self.conv7_8 = nn.Sequential(
            ConvBlock(512, 512, stride=1),
            ConvBlock(512, 512, stride=1)  # Output: 19x19x512
        )
            
        self.conv9_11 = nn.Sequential(
            # conv 9
            ConvBlock(512, 512, stride=2),
            # conv 10
            ConvBlock(512, 512, stride=1),
            # conv 11
            ConvBlock(512, 512, stride=1)
        )
        
        self.conv12_13 = nn.Sequential(
            ConvBlock(512, 512, stride=2),
            ConvBlock(512, 1024, stride=1),
        )
        

    def forward(self, x):
        x = self.img_6(x)
        c8 = self.conv7_8(x)
        c11 = self.conv9_11(c8)
        c13 = self.conv12_13(c11)
        
        return c8, c11, c13
    

class HandDetectionNetwork(nn.Module):
    def __init__(self, num_anchors=6):
        super(HandDetectionNetwork, self).__init__()
        self.backbone = ModifiedMobileNet()
        

        
        self.extra_14 = nn.Sequential(
            # Pointwise pentru reducerea canalelor (eficiență)
            nn.Conv2d(1024, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Depthwise + Pointwise pentru a ajunge la 512 canale și rezoluție 5x5
            ConvBlock(256, 512, stride=2) 
        )
        # Exemplu de Extra Layer (conv_15_2)
        self.extra_15 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ConvBlock(128, 256, stride=2)
        )
        self.extra_16 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 128, stride=2, padding = 0)
        )
        
        # În clasa HandTracker, secțiunea de inițializare a blocurilor de fuziune:

        self.fusion_15 = TopDownFusionBlock(channels_low=256,  channels_high_star=128)  # Out: 128
        self.fusion_14 = TopDownFusionBlock(channels_low=512,  channels_high_star=128)  # Out: 128
        self.fusion_13 = TopDownFusionBlock(channels_low=1024, channels_high_star=128)  # Out: 128
        self.fusion_11 = TopDownFusionBlock(channels_low=512,  channels_high_star=128)  # Out: 128
        self.fusion_8  = TopDownFusionBlock(channels_low=512,  channels_high_star=128)  # Out: 128
        
        self.num_anchors = num_anchors
        
        def build_heads(in_channels):
            return nn.ModuleDict({
                'cls': nn.Conv2d(in_channels, num_anchors, kernel_size=3, padding=1), 
                'loc': nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)  
            })

        self.heads_8 = build_heads(128)
        self.heads_11 = build_heads(128)
        self.heads_13 = build_heads(128)
        self.heads_14 = build_heads(128)
        self.heads_15 = build_heads(128)
        self.heads_16 = build_heads(128)

    def forward(self, x):
        
        c8, c11, c13 = self.backbone(x)
        
        c14 = self.extra_14(c13)
        c15 = self.extra_15(c14)
        c16 = self.extra_16(c15)
        
        # Fuziune succesivă
        c16_star = c16 
    
        # Rezoluția crește, informația de context coboară
        c15_star = self.fusion_15(c15, c16_star) # Rezultat: 3x3 actualizat
        c14_star = self.fusion_14(c14, c15_star) # Rezultat: 5x5 actualizat
        c13_star = self.fusion_13(c13, c14_star) # Rezultat: 10x10 actualizat
        c11_star = self.fusion_11(c11, c13_star) # Rezultat: 19x19 actualizat
        c8_star  = self.fusion_8(c8, c11_star)   # Rezultat: 38x38 actualizat
        
        features = [c8_star, c11_star, c13_star, c14_star, c15_star, c16_star]
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
    def __init__(self, anchors_cxcy, threshold=0.5, weight_box=1.0): # Scădem weight_box la 1.0 conform lucrării
        super(HandTrackerLoss, self).__init__()
        self.threshold = threshold
        self.weight_box = weight_box
        self.priors_cxcy = anchors_cxcy
        self.priors_xy = cxcy_to_xy(anchors_cxcy)

    def forward(self, out, labels):
        # 1. Pregătirea predicțiilor
        preds_cls = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1) 
                               for o in out["cls"]], dim=1) 
        preds_loc = torch.cat([o.permute(0, 2, 3, 1).contiguous().view(o.size(0), -1, 4) 
                               for o in out["loc"]], dim=1) 

        batch_size = preds_cls.size(0)
        num_priors = self.priors_cxcy.size(0)
        device = preds_cls.device
        
        target_conf = torch.zeros(batch_size, num_priors, device=device)
        target_loc = torch.zeros(batch_size, num_priors, 4, device=device)

        # 2. Matching (SSD logic)
        for idx in range(batch_size):
            truths = labels[idx][:, 1:] 
            if truths.size(0) == 0: continue
                
            overlaps = box_iou(truths, self.priors_xy.to(truths.device))
            best_truth_overlap, best_truth_idx = overlaps.max(0)
            
            matched_truths = truths[best_truth_idx]
            target_loc[idx] = encode_boxes(matched_truths, self.priors_cxcy.to(device))
            target_conf[idx][best_truth_overlap > self.threshold] = 1.0

        # 3. HARD NEGATIVE MINING
        # Calculăm loss-ul brut pentru clasificare
        loss_c = F.binary_cross_entropy_with_logits(preds_cls, target_conf, reduction='none')
        
        pos_mask = target_conf > 0  # Unde avem mâini
        num_pos = pos_mask.long().sum(1, keepdim=True)

        # Sortăm doar negativele (fundalul) după eroare
        loss_c_for_mining = loss_c.clone()
        loss_c_for_mining[pos_mask] = 0 # Ignorăm pozitivele la sortare
        
        _, loss_idx = loss_c_for_mining.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        
        # Selectăm de 3 ori mai multe negative decât pozitive 
        num_neg = torch.clamp(3 * num_pos, max=num_priors - 1)
        hard_neg_mask = idx_rank < num_neg
        
        # 4. CALCULUL LOSS-URILOR FINALE
        # Confidență: Calculată pe Pozitive + Hard Negatives
        conf_mask = pos_mask | hard_neg_mask
        loss_conf = loss_c[conf_mask].sum() / (num_pos.sum() + 1e-5)
        
        # Localizare: Calculată DOAR pe Pozitive
        loss_box_raw = F.smooth_l1_loss(preds_loc, target_loc, reduction='none').sum(dim=-1)
        loss_box = (loss_box_raw * pos_mask).sum() / (num_pos.sum() + 1e-5)

        # Total loss conform structurii simplificate a lucrării 
        return loss_conf + (self.weight_box * loss_box)