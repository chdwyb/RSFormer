import torch
import random
import torch.nn.functional as F


# calculate psnr
def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps


# set seeds
def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


# pad
def pad(x, factor=16, mode='reflect'):
    _, _, h_even, w_even = x.shape
    padh_left = (factor - h_even % factor) // 2
    padw_top = (factor - w_even % factor) // 2
    padh_right = padh_left if h_even % 2 == 0 else padh_left + 1
    padw_bottom = padw_top if w_even % 2 == 0 else padw_top + 1
    x = F.pad(x, pad=[padw_top, padw_bottom, padh_left, padh_right], mode=mode)
    return x, (padh_left, padh_right, padw_top, padw_bottom)


# reverse pad
def unpad(x, pad_size):
    padh_left, padh_right, padw_top, padw_bottom = pad_size
    _, _, newh, neww = x.shape
    h_start = padh_left
    h_end = newh - padh_right
    w_start = padw_top
    w_end = neww - padw_bottom
    x = x[:, :, h_start:h_end, w_start:w_end]
    return x
