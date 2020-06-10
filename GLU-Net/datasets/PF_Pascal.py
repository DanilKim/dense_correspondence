import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import os
    
# some parts of codes are from 'https://github.com/ignacio-rocco/weakalign'
class PF_Pascal(Dataset):
    def __init__(self, csv_path, image_path, feature_H, feature_W, eval_type='image_size'):
        self.feature_H = feature_H
        self.feature_W = feature_W
        
        self.image_H = (self.feature_H-2) * 16
        self.image_W = (self.feature_W-2) * 16
        
        self.data_info = pd.read_csv(csv_path)
        
        self.transform = transforms.Compose([transforms.Resize((self.image_H, self.image_W), interpolation=2),
                                              transforms.Pad(16), # pad zeros around borders to avoid boundary artifacts
                                              transforms.ToTensor()])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_A_names = self.data_info.iloc[:, 0]
        self.image_B_names = self.data_info.iloc[:, 1]
        self.class_num = self.data_info.iloc[:, 2]
        self.point_A_coords = self.data_info.iloc[:, 3:5]
        self.point_B_coords = self.data_info.iloc[:, 5:7]        
        self.L_pck = self.data_info.iloc[:,7].values.astype('float') # L_pck of source
        self.image_path = image_path
        self.eval_type = eval_type

    def get_image(self, image_name_list, idx):
        image_name = os.path.join(self.image_path, image_name_list[idx])
        image = Image.open(image_name)
        width, height = image.size
        return image, torch.FloatTensor([height, width])

    def get_points(self, point_coords_list, idx):
        X = np.fromstring(point_coords_list.iloc[idx, 0], sep=';')
        Y = np.fromstring(point_coords_list.iloc[idx, 1], sep=';')
        point_coords = np.concatenate((X.reshape(1, len(X)), Y.reshape(1, len(Y))), axis=0)
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords


    def __getitem__(self, idx):
        # get pre-processed images
        image1, image1_size = self.get_image(self.image_A_names, idx)
        image2, image2_size = self.get_image(self.image_B_names, idx)
        class_num = int(self.class_num[idx])-1
        image1_var = self.transform(image1)
        image2_var = self.transform(image2)

        # get pre-processed point coords
        point_A_coords = self.get_points(self.point_A_coords, idx)
        point_B_coords = self.get_points(self.point_B_coords, idx)
        # compute PCK reference length L_pck (equal to max bounding box side in image_B)
        if self.eval_type == 'bounding_box':
            # L_pck = torch.FloatTensor([torch.max(point_B_coords.max(1)[0] - point_B_coords.min(1)[0])]) # for PF WILLOW
            L_pck = torch.FloatTensor(np.fromstring(self.L_pck[idx]).astype(np.float32)) # max(h,w), where h&w are height&width of bounding-box provided by Pascal dataset
        elif self.eval_type == 'image_size':
            N_pts = torch.sum(torch.ne(point_A_coords[0,:],-1))
            point_A_coords[0,0:N_pts] = point_A_coords[0,0:N_pts] * self.image_W / image1_size[1] # rescale x coord.
            point_A_coords[1,0:N_pts] = point_A_coords[1,0:N_pts] * self.image_H / image1_size[0] # rescale y coord.
            point_B_coords[0,0:N_pts] = point_B_coords[0,0:N_pts] * self.image_W / image2_size[1] # rescale x coord.
            point_B_coords[1,0:N_pts] = point_B_coords[1,0:N_pts] * self.image_H / image2_size[0] # rescale y coord.
            image1_size = torch.FloatTensor([self.image_H,self.image_W])
            image2_size = torch.FloatTensor([self.image_H,self.image_W])
            L_pck = torch.FloatTensor([self.image_H]) if self.image_H >= self.image_W else torch.FloatTensor([self.image_W])
        else:
            raise ValueError('Invalid eval_type')

        return {
        'image1_rgb': transforms.ToTensor()(image1), 
        'image2_rgb': transforms.ToTensor()(image2),
        'image1': self.normalize(image1_var), 
        'image2': self.normalize(image2_var),
        'image1_points': point_A_coords, 
        'image2_points': point_B_coords, 
        'L_pck': L_pck,
        'image1_size': image1_size, 
        'image2_size': image2_size, 
        'class_num':class_num
        }

    def __len__(self):
        return len(self.data_info.index)

