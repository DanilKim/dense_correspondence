
import os
import numpy as np
import argparse
import random
from utils.io import boolean_string
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets.training_dataset import HomoAffTps_Dataset
from utils.pixel_wise_mapping import remap_using_flow_fields
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
from utils.image_transforms import ArrayToTensor
from utils.io import writeFlow


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='DGC-Net train script')
    parser.add_argument('--image_data_path', type=str,
                        help='path to directory containing the original images.')
    parser.add_argument('--dataset', type=str, choices=['DPED_CityScape_ADE', 'Sintel'],
                        help='Which dataset to process and save')
    parser.add_argument('--split', type=str, choices=['train', 'val'],
                        help='dataset split')
    parser.add_argument('--csv_dir', type=str, default='datasets/csv_files',
                        help='directory to the CSV files')
    parser.add_argument('--save_dir', type=str, default='datasets/training_datasets',
                        help='path directory to save the image pairs and corresponding ground-truth flows')
    parser.add_argument('--plot', default=False, type=boolean_string,
                        help='plot as examples the first 4 pairs ? default is False')
    parser.add_argument('--seed', type=int, default=1981,
                        help='Pseudo-RNG seed')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args.csv_path = os.path.join(args.csv_dir, 'homo_aff_tps_' + args.split + '_' + args.dataset + '.csv')
    args.save_dir = os.path.join(args.save_dir, args.dataset, args.split)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    source_flow = (args.dataset == 'Sintel')

    image_dir = os.path.join(args.save_dir, 'images')
    flow_dir = os.path.join(args.save_dir, 'flow')
    mask_dir = os.path.join(args.save_dir, 'mask')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    # datasets
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    pyramid_param = [520]

    # training dataset
    train_dataset = HomoAffTps_Dataset(image_path=args.image_data_path,
                                       csv_file=args.csv_path,
                                       transforms=source_img_transforms,
                                       transforms_target=target_img_transforms,
                                       pyramid_param=pyramid_param,
                                       source_flow=source_flow,
                                       get_flow=True,
                                       output_size=(520, 520))

    test_dataloader = DataLoader(train_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1)

    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, minibatch in pbar:
        image_source = minibatch['source_image']  # shape is 1x3xHxW
        image_target = minibatch['target_image']
        if image_source.shape[1] == 3:
            image_source = image_source.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            image_source = image_source[0].numpy().astype(np.uint8)

        if image_target.shape[1] == 3:
            image_target = image_target.permute(0, 2, 3, 1)[0].numpy().astype(np.uint8)
        else:
            image_target = image_target[0].numpy().astype(np.uint8)

        flow_gt = minibatch['flow_map'][0].permute(1, 2, 0).numpy()  # now shape is HxWx2

        # save the flow file and the images files
        base_name = 'image_{}'.format(i)
        name_flow = base_name + '_flow.flo'
        writeFlow(flow_gt, name_flow, flow_dir)
        imageio.imwrite(os.path.join(args.save_dir, 'images/', base_name + '_img_1.jpg'), image_source)
        imageio.imwrite(os.path.join(args.save_dir, 'images/', base_name + '_img_2.jpg'), image_target)
        if source_flow:
            image_mask = minibatch['correspondence_mask'][0]
            image_mask = ((1 - image_mask) * 255).numpy().astype(np.uint8)
            imageio.imwrite(os.path.join(args.save_dir, 'mask/', base_name + '_occlusion.png'), image_mask)

        # plotting to make sure that everything is working
        if args.plot and i < 4:
            # just for now
            fig, axis = plt.subplots(1, 3, figsize=(20, 20))
            axis[0].imshow(image_source)
            axis[0].set_title("Image source")
            axis[1].imshow(image_target)
            axis[1].set_title("Image target")
            remapped_gt = remap_using_flow_fields(image_source, flow_gt[:, :, 0], flow_gt[:, :, 1])

            axis[2].imshow(remapped_gt)
            axis[2].set_title("Warped source image according to ground truth flow")
            fig.savefig(os.path.join(args.save_dir, 'synthetic_pair_{}'.format(i)), bbox_inches='tight')
            plt.close(fig)

