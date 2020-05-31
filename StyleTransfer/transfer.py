import torch
import argparse
import torchvision.models as models
from model import run_style_transfer
import imageio
import os.path
import numpy as np
from tools import sintel_clean, TSS_Dataset, image_loader
import glob
from tqdm import tqdm
        

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, default=None, help='Path for target dataset to be transfered')
    parser.add_argument('--source_path', type=str, default=None, help='Path for source dataset(style)')
    parser.add_argument('--save_dir', type=str, default='./save', help='Path for save path')
    args = parser.parse_args()
    print("Loading content image sources...")
    content_datasets = TSS_Dataset(args.target_path) # dictionary

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    print("Style transfering...")
    cnt = 0
    for name, dataset in content_datasets.items():
        for cnt, one_img in enumerate(dataset):
            print("[",name,"] : ")
            Folder_path, File_name = os.path.split(dataset.get_path(cnt))
            Split_Folder = Folder_path.split('/')
            print(Folder_path, File_name)

            target_dir = os.path.join(args.save_dir, Split_Folder[-3], Split_Folder[-2], Split_Folder[-1])
            print(target_dir)
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)

            content_img = one_img.to(device, torch.float)
            style_images = sintel_clean(args.source_path, content_img.shape, device)
            output = run_style_transfer(
                                        cnn, cnn_normalization_mean, cnn_normalization_std,
                                        content_img, style_images, content_img, device=device
                                        )
            result = output.squeeze().permute(1,2,0).detach().cpu().numpy()
            imageio.imwrite(os.path.join(target_dir, File_name), result)

    
    