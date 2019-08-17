import numpy as np

import torch

def visualize_tensor(common_tensors, mean, div):
    '''
        common_tensors: list of tensors, follow the same mean / div with image.
    '''
    together = []
    for ct in common_tensors:
        ct = unormalize(ct.detach().cpu(), mean, div)
        if ct.max().item() <= 1:
            ct *= 255
        together.append(ct)
    together = torch.cat(together, dim=3)
    together = together.permute(1,0,2,3).contiguous()
    together = together.view(together.size(0), -1, together.size(3))
    return together

def unormalize(tensor, mean, div):
    for c, (m, d) in enumerate(zip(mean, div)):
        tensor[:,c,:,:].mul_(d).add_(m)
    return tensor
