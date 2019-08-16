import numpy as np

import torch

def visualize_tensor(rgb, mask, eraser, target, common_tensors):
    '''
        rgb: N3HW
        mask: N1HW
        eraser: N1HW
        target: NHW
        ct: N1HW
    '''
    comb = (mask > 0).float()
    comb[eraser == 1] = 0.5

    together = [rgb.detach().cpu() * 255,
                (mask > 0).float().repeat(1,3,1,1).detach().cpu() * 255,
                comb.repeat(1,3,1,1).detach().cpu() * 255]
    for ct in common_tensors:
        together.append(ct.detach().cpu().repeat(1,3,1,1) * 255)
    target = target.detach().cpu().float()
    if target.max().item() == 255: # ignore label
        target[target == 255] = 0.5
    together.append(target.unsqueeze(1).repeat(1,3,1,1) * 255)
    together = torch.cat(together, dim=3)
    together = together.permute(1,0,2,3).contiguous()
    together = together.view(together.size(0), -1, together.size(3))
    return together

