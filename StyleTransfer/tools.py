import os.path
import glob
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import torch
from imageio import imread
import cv2
import torch.utils.data as data


unloader = transforms.ToPILImage()  # reconvert into PIL image
loader = transforms.Compose([
        transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(root, path_imgs, shape=None):
    imgs = os.path.join(root, path_imgs)
    imgs = imread(imgs).astype(np.uint8) 
    if shape is not None:
        imgs = cv2.resize(imgs, dsize=(shape[3], shape[2]))
    return imgs

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, shape):
        self.root = root
        self.path_list = path_list
        self.shape = shape
        self.loader = image_loader
        self.source_image_transform = transforms.Compose([transforms.ToTensor()])
    
    def get_path(self, idx):
        return os.path.abspath(self.path_list[idx])

    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs = self.path_list[index]
        inputs = self.loader(self.root, inputs, self.shape)
        inputs = self.source_image_transform(inputs)
        return inputs.unsqueeze(0)
                
    def __len__(self):
        return len(self.path_list)


def make_dataset_sintel(dataset_dir):
    assert(os.path.isdir(os.path.join(dataset_dir)))
    image_list = []
    for scene_path in sorted(glob.glob(os.path.join(dataset_dir,'*'))):
        scene_name_only = os.path.relpath(scene_path,os.path.join(dataset_dir))
        for images in sorted(glob.glob(os.path.join(scene_path, '*'))):
            image_list.append(images)
    return image_list

def make_dataset_TSS(dataset_dir):
    assert(os.path.isdir(os.path.join(dataset_dir)))
    image_list = []
    for scene_path in sorted(glob.glob(os.path.join(dataset_dir,'*'))):
        scene_name_only = os.path.relpath(scene_path,os.path.join(dataset_dir))
        for images in sorted(glob.glob(os.path.join(scene_path, 'image?.png'))):
            image_list.append(images)
    return image_list

def sintel_clean(source_path, shape, device):
    print("Loading style image sources...")
    source_datapath = make_dataset_sintel(source_path + 'clean')
    dataset = ListDataset(root='./', path_list = source_datapath, shape=shape)
    return dataset


def TSS_Dataset(target_path):
    target_1 = make_dataset_TSS(target_path + 'FG3DCar')
    dataset1 = ListDataset(root='./', path_list = target_1, shape=None)

    target_2 = make_dataset_TSS(target_path + 'JODS')
    dataset2 = ListDataset(root='./', path_list = target_2, shape=None)

    target_3 = make_dataset_TSS(target_path + 'PASCAL')
    dataset3 = ListDataset(root='./', path_list = target_3, shape=None)

    return {'FG3DCar' : dataset1, 'JODS' : dataset2, 'PASCAL' : dataset3}

