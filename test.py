import os
import sys
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from model import SASCFormer
from utils import pad, unpad
from datasets import MyTestDataSet
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def test(args):
    # model
    model_restoration = SASCFormer().cuda() if args.cuda else SASCFormer()
    # multi-gpu training
    model_restoration = nn.DataParallel(model_restoration)
    # test dataset
    path_val_input, path_val_target = args.val_data + '/input/', args.val_data + '/target/'
    datasetTest = MyTestDataSet(path_val_input, path_val_target)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=args.num_works, pin_memory=True)

    # load model
    if args.cuda:
        model_restoration.load_state_dict(torch.load(args.resume_state))
    else:
        model_restoration.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))
    model_restoration.eval()

    # testing
    with torch.no_grad():
        for index, (x, y, name) in enumerate(tqdm(testLoader, desc='Test !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_, target = (x.cuda(), y.cuda()) if args.cuda else (x, y)
            input_test, pad_size = pad(input_test, factor=args.pad_factor)  # pad
            restored_ = model_restoration(input_)
            restored_ = unpad(restored_, pad_size)  # unpad

            if args.result_save:
                save_image(restored_, os.path.join(args.result_dir, name[0]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data', type=str, default='./RSCityscapes/test')
    parser.add_argument('--result_dir', type=str, default='./RSCityscapes/test/result/')
    parser.add_argument('--resume_state', type=str, default='./models/RSCityscapes.pth')
    parser.add_argument('--pad_factor', type=int, default=16, help='expand input to a multiplier pf pad_factor')
    parser.add_argument('--result_save', type=bool, default=True, help='to save the result')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--num_works', type=int, default=4)
    args = parser.parse_args()

    test(args)
