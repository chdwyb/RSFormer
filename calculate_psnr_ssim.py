import os
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config import Options

opt = Options()
path_result = opt.Result_Path_Test
path_target = opt.Target_Path_Test
image_list = os.listdir(path_target)
L = len(image_list)
print(L)
psnr, ssim = 0, 0

for i in range(L):
    image_in = cv2.imread(path_result+str(image_list[i]), 1)
    image_tar = cv2.imread(path_target+str(image_list[i]), 1)
    ps = peak_signal_noise_ratio(image_in, image_tar)
    ss = structural_similarity(image_in/255., image_tar/255., channel_axis=-1)
    psnr += ps
    ssim += ss
    print(i)
print(psnr/L, ssim/L)