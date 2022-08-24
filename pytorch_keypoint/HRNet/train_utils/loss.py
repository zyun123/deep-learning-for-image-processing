import torch
import torch.nn.functional as F

class KpLoss(object):
    def __init__(self,use_mse=True):
        self.use_mse = use_mse
        self.criterion = torch.nn.MSELoss(reduction='none')
        # self.cross_entropy = torch.nn.functional.cross_entropy

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs,num_kps,H,W = logits.shape
        if self.use_mse:
            # [num_kps, H, W] -> [B, num_kps, H, W]
            heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
            # [num_kps] -> [B, num_kps]
            kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])

            # [B, num_kps, H, W] -> [B, num_kps]
            loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
            loss = torch.sum(loss * kps_weights) / bs
        else:
            logits = logits.view(bs*num_kps,H*W)
            keypoint_targets = torch.hstack([t["heatmap"] for t in targets]).to(device)
            loss  = F.cross_entropy(logits,keypoint_targets)


        return loss
