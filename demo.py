import argparse
from datasets import *
from utils import pad, unpad
from model import SASCFormer
from torchvision.utils import save_image


def test_single_image(args):
    # model
    model_restoration = SASCFormer().cuda() if args.cuda else SASCFormer()
    # input image
    image_input = Image.open(args.image_path).convert('RGB')
    image_input = ttf.to_tensor(image_input).unsqueeze(0)
    # testing
    with torch.no_grad():
        torch.cuda.empty_cache()

        image_input = image_input.cuda() if args.cuda else image_input

        image_input, pad_size = pad(image_input, factor=args.pad_factor)  # pad
        restored = model_restoration(image_input)
        restored = unpad(restored, pad_size)  # unpad

        if args.result_save:
            save_image(restored, args.result_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./RSCityscapes/test/input/0001.png')
    parser.add_argument('--result_path', type=str, default='./RSCityscapes/test/result/0001.png')
    parser.add_argument('--resume_state', type=str, default='./models/RSCityscapes.pth')
    parser.add_argument('--pad_factor', type=int, default=16, help='expand input to a multiplier pf pad_factor')
    parser.add_argument('--result_save', type=bool, default=True, help='to save the result')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--num_works', type=int, default=4)
    args = parser.parse_args()

    test_single_image(args)
