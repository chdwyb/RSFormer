import os
import sys
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from model import SASCFormer
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import torchPSNR, setseed
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import MyTrainDataSet, MyValueDataSet
from losses import FocalFrequencyLoss, CharbonnierLoss


def train(args):

    cudnn.benchmark = True
    setseed(args.seed)  # set seeds

    # model
    model_restoration = SASCFormer().cuda() if args.cuda else SASCFormer()
    # multi-gpu training
    device_ids = [i for i in range(torch.cuda.device_count())]
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)
    # optimizer
    optimizer = optim.Adam(model_restoration.parameters(), lr=args.lr)
    # scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr_min)
    # training dataset
    path_train_input, path_train_target = args.train_data + '/input/', args.train_data + '/target/'
    datasetTrain = MyTrainDataSet(path_train_input, path_train_target, patch_size=args.patch_size_train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=args.batch_size_train, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    # validation dataset
    path_val_input, path_val_target = args.val_data + '/input/', args.val_data + '/target/'
    datasetValue = MyValueDataSet(path_val_input, path_val_target, patch_size=args.patch_size_val)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=args.batch_size_val, shuffle=True,
                             drop_last=True, num_workers=args.num_works, pin_memory=True)
    # load pre model
    if os.path.exists(args.resume_state):
        if args.cuda:
            model_restoration.load_state_dict(torch.load(args.resume_state))
        else:
            model_restoration.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))

    frequency_loss = FocalFrequencyLoss(loss_weight=0.4)
    spatial_loss = CharbonnierLoss()
    if args.cuda:
        frequency_loss = frequency_loss.cuda()
        spatial_loss = spatial_loss.cuda()

    scaler = GradScaler()
    best_psnr = 0
    for epoch in range(args.epoch):
        model_restoration.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        # train
        for index, (x, y) in enumerate(iters, 0):

            model_restoration.zero_grad()
            optimizer.zero_grad()

            input_train = Variable(x).cuda() if args.cuda else Variable(x)
            target_train = Variable(y).cuda() if args.cuda else Variable(y)

            with autocast(args.autocast):
                restored_train = model_restoration(input_train)
                loss = spatial_loss(restored_train, target_train) + frequency_loss(restored_train, target_train)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epochLoss += loss.item()
            iters.set_description('Train !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, args.Epoch, loss.item()))
        # validation
        if epoch % args.val_frequency == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_val, target_val = (x.cuda(), y.cuda()) if args.cuda else (x, y)
                with torch.no_grad():
                    restored_val = model_restoration(input_val)
                for restored_val, target_val in zip(restored_val, target_val):
                    psnr_val_rgb.append(torchPSNR(restored_val, target_val))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            # save the best model
            if psnr_val_rgb >= best_psnr:
                best_psnr = psnr_val_rgb
                torch.save(model_restoration.state_dict(), args.save_state)
            print("----------------------------------------------------------------------------------------------")
            print("Validation Finished, Current PSNR: {:.4f}, Best PSNR: {:.4f}.".format(psnr_val_rgb, best_psnr))
            print("----------------------------------------------------------------------------------------------")
        scheduler.step()
    print("Training Process Finished !")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size_train', type=int, default=32)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--patch_size_train', type=int, default=256)
    parser.add_argument('--patch_size_val', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_min', type=float, default=1e-7)
    parser.add_argument('--train_data', type=str, default='./RSCityscapes/train')
    parser.add_argument('--val_data', type=str, default='./RSCityscapes/val')
    parser.add_argument('--resume_state', type=str, default='./model_resume.pth')
    parser.add_argument('--save_state', type=str, default='./model_best.pth')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--val_frequency', type=int, default=3)
    parser.add_argument('--loss_weight', type=float, default=0.04)
    parser.add_argument('--autocast', type=bool, default=True)
    parser.add_argument('--num_works', type=int, default=4)
    args = parser.parse_args()

    train(args)




