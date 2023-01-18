import sys
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

from RSFormer import RSFormer
from datasets import *
from config import Options


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


if __name__ == '__main__':

    opt = Options()

    inputPathTest = opt.Input_Path_Test
    resultPathTest = opt.Result_Path_Test

    myNet = RSFormer()
    # myNet = nn.DataParallel(myNet)
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    datasetTest = MyTestDataSet(inputPathTest)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True)

    print('--------------------------------------------------------------')
    # pretrained model
    if opt.CUDA_USE:
        myNet.load_state_dict(torch.load('./model_best.pth'))
    else:
        myNet.load_state_dict(torch.load('./model_best.pth', map_location=torch.device('cpu')))
    myNet.eval()

    with torch.no_grad():
        timeStart = time.time()
        for index, (x, name) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_test = x.cuda() if opt.CUDA_USE else x

            input_test, pad_size = pad(input_test, factor=16)
            output_test = myNet(input_test)
            output_test = unpad(output_test, pad_size)

            save_image(output_test, resultPathTest + name[0])
        timeEnd = time.time()
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
