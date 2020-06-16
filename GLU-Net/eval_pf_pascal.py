import torch
import torch.nn.functional as F
import numpy as np
import os
import random
import argparse
from models.models_compared import GLOCAL_Net, GLU_Net
from utils.evaluate import calculate_epe_and_pck_per_dataset
import json
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import PF_Pascal
from utils.image_transforms import ArrayToTensor
from tqdm import tqdm
from utils.io import writeFlow
import torch.nn as nn

parser = argparse.ArgumentParser(description="GLUNET evaluation on PF-PASCAL")
parser.add_argument('--pre_trained_models_dir', type=str, default='pre_trained_models/',
                    help='Directory containing the pre-trained-models.')
parser.add_argument('--save', default=True, type=bool,
                    help='save the flow files ? default is False')
parser.add_argument('--save_dir', type=str, default='pf-pascal-eval',
                    help='path to directory to save the text files and results')                    
parser.add_argument('--feature_h', type=int, default=20, help='height of feature volume')
parser.add_argument('--feature_w', type=int, default=20, help='width of feature volume')
parser.add_argument('--test_csv_path', type=str, default='GLUNet_data/testing_datasets/PF-dataset-PASCAL/PF-dataset-PASCAL/bbox_test_pairs_pf_pascal.csv', help='directory of test csv file')
parser.add_argument('--test_image_path', type=str, default='GLUNet_data/testing_datasets/PF-dataset-PASCAL', help='directory of test data')
parser.add_argument('--eval_type', type=str, default='image_size', choices=('bounding_box','image_size'), help='evaluation type for PCK threshold (bounding box | image size)')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# PCK metric from 'https://github.com/ignacio-rocco/weakalign/blob/master/util/eval_util.py'
def correct_keypoints(source_points, warped_points, L_pck, alpha=0.1):
    # compute correct keypoints
    p_src = source_points[0,:]
    p_wrp = warped_points[0,:]

    N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
    point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
    L_pck_mat = L_pck[0].expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck

# Data Loader
print("Instantiate dataloader")
test_dataset = PF_Pascal(args.test_csv_path, args.test_image_path, args.feature_h, args.feature_w, args.eval_type)
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False, num_workers = 1)

# Instantiate model
print("Instantiate model")
#net = SFNet(args.feature_h, args.feature_w, beta=args.beta, kernel_sigma = args.kernel_sigma)
net = GLU_Net(model_type='Sintel',
              path_pre_trained_models=args.pre_trained_models_dir,
              consensus_network=False,
              cyclic_consistency=True,
              iterative_refinement=True,
              apply_flipping_condition=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold_range = np.linspace(0.002, 0.2, num=50)

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
per_class_pck = np.zeros(20)
num_instances = np.zeros(20)
with torch.no_grad():
    print('Computing PCK@Test set...')
    total_correct_points = 0
    total_points = 0
    for i, batch in tqdm(enumerate(test_loader)):
        source_image = batch['image1'].to(device)
        target_image = batch['image2'].to(device)

        src_image_H = int(batch['image1_size'][0][0])
        src_image_W = int(batch['image1_size'][0][1])
        tgt_image_H = int(batch['image2_size'][0][0])
        tgt_image_W = int(batch['image2_size'][0][1])

        grid_X, grid_Y = np.meshgrid(np.linspace(-1, 1, tgt_image_W), np.linspace(-1, 1, tgt_image_H))
        grid_X = torch.tensor(grid_X, dtype=torch.float, requires_grad=False).to(device)
        grid_Y = torch.tensor(grid_Y, dtype=torch.float, requires_grad=False).to(device)
        grid_XY = torch.cat((grid_X.view(1,tgt_image_H,tgt_image_W,1), grid_Y.view(1,tgt_image_H,tgt_image_W,1)),3)

        # get a flow of target to source // 
        flow_T2S = net.estimate_flow(target_image, source_image, device, mode='channel_first')

        if (args.visualize):
            from utils.pixel_wise_mapping import remap_using_flow_fields
            import matplotlib.pyplot as plt
            resized_target = F.interpolate(target_image, size=(flow_T2S.shape[2], flow_T2S.shape[3]), mode='bilinear',
                                           align_corners=True)
            warped_source_image = remap_using_flow_fields(resized_target.squeeze().permute(1, 2, 0).cpu().numpy(),
                                                          flow_T2S.squeeze()[0].cpu().numpy(),
                                                          flow_T2S.squeeze()[1].cpu().numpy())

            fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(30, 30))
            axis1.imshow(resized_target.squeeze().permute(1, 2, 0).cpu().numpy())
            axis1.set_title('Target image')
            axis2.imshow(raw_src)
            axis2.set_title('Source image')
            axis3.imshow(warped_source_image)
            axis3.set_title('Warped target image according to estimated T2S_flow by GLU-Net')
            #fig.savefig(os.path.join(args.write_dir, 'Warped_' + tgt_name + '_to_' + src_name + '.png'),
            #            bbox_inches='tight')
            plt.close(fig)

        grid_warped = F.grid_sample(target_image, flow_T2S.permute(0, 2, 3, 1)).to(device)
        warped_target = F.grid_sample((batch['image2_rgb']).to(device), flow_T2S.permute(0, 2, 3, 1))
        grid = F.interpolate(grid_warped, size = (tgt_image_H,tgt_image_W), mode='bilinear', align_corners=True)
        #grid = F.interpolate(flow_T2S, size=(tgt_image_H, tgt_image_W), mode='bilinear', align_corners=True)
        grid = grid.permute(0,2,3,1)
        # grid = grid + grid_XY
        grid_np = grid.cpu().data.numpy()

        image1_points = batch['image1_points'][0]
        image2_points = batch['image2_points'][0]

        est_image1_points = np.zeros((2,image1_points.size(1)))
        for j in range(image2_points.size(1)):
            point_x = int(np.round(image2_points[0,j]))
            point_y = int(np.round(image2_points[1,j]))

            if point_x == -1 and point_y == -1:
                continue

            if point_x == tgt_image_W:
                point_x = point_x - 1

            if point_y == tgt_image_H:
                point_y = point_y - 1

            est_y = (grid_np[0,point_y,point_x,1] + 1)*(src_image_H-1)/2
            est_x = (grid_np[0,point_y,point_x,0] + 1)*(src_image_W-1)/2
            est_image1_points[:,j] = [est_x,est_y]

        total_correct_points += correct_keypoints(batch['image1_points'], torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'], alpha=0.1)
        per_class_pck[batch['class_num']] += correct_keypoints(batch['image1_points'], torch.FloatTensor(est_image1_points).unsqueeze(0), batch['L_pck'], alpha=0.1)
        num_instances[batch['class_num']] += 1
    PCK = total_correct_points / len(test_dataset)
    print('PCK: %5f' % PCK)
    per_class_pck = per_class_pck / num_instances

    for i in range(per_class_pck.shape[0]):
        print('%-12s' % class_names[i],': %5f' % per_class_pck[i])