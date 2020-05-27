import numpy as np
import cv2
from utils.pixel_wise_mapping import remap_using_correspondence_map
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys



def collate_fn_make_same_size(batch):
    H_source = sys.maxsize
    W_source = sys.maxsize
    for sample in batch:
        Height = sample['source_image'].shape[1]
        Width = sample['source_image'].shape[2]
        if(Height <= H_source): H_source = Height
        if(Width <= W_source): W_source = Width
    New_source=[]
    New_target=[]
    New_mask=[]
    New_flow=[]
    New_size=[]

    for sample in batch:
        source = sample['source_image'][:, :int(H_source), :int(W_source)]
        target = sample['target_image'][:, :int(H_source), :int(W_source)]
        mask = sample['correspondence_mask'][:int(H_source), :int(W_source)]
        flow = sample['flow_map'][:, :int(H_source), :int(W_source)]

        New_source.append(source)
        New_target.append(target)
        New_mask.append(mask)
        New_flow.append(flow)
        New_size.append(source.shape)

    return {
        'source_image': torch.stack(New_source),
        'target_image': torch.stack(New_target),
        'flow_map': torch.stack(New_flow),
        'correspondence_mask': torch.stack(New_mask),
        'source_image_size': New_size
    }




def get_path_from_dataloader(dataloader):
    if not isinstance(dataloader, tuple):
        raise "Argument must be in 'Tuple' form."
    train_dataset = dataloader[0]
    test_dataset = dataloader[1]
    print(train_dataset.get_path())

    return train_dataset.get_path(), test_dataset.get_path()


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)

    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D

def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples

def random_crop(img, size, seed, value=[0,0,0]):
    if isinstance(size, tuple):
        size = size[0]
        #size is W,H
    
    img = img.copy()
    
    h = img.shape[0]
    w = img.shape[1]
    
    pad_w = 0
    pad_h = 0

    if w < size:
        pad_w = np.int(np.ceil((size - w) / 2))
    if h < size:
        pad_h = np.int(np.ceil((size - h) / 2))

    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=value)
    h, w = img_pad.shape[:2]
    random.seed(seed)
    x = random.randint(0, w - size)
    if(x%2 == 1): # if width len is odd number, make it even number.
        if((x-1) <= 0) : x = x + 1
        elif((x+1) > (w-size)) : x = x-1
    if (x % 8) >= 1 :  # if width len is not correctly divided by eight, make it correct.
        if((x) + (8 - x%8) > w-size) : x = x - (x%8)
        else : x = x + (8 - x%8)
    
    #y = random.randint(0, h - size)
    y = 0

    if len(img.shape)==3 :
        imgr = img_pad[y:y+h, x:x+size, :]
    else: 
        imgr = img_pad[y:y+h, x:x+size]
    return imgr
    
def resize(img, size):
    result = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    return np.asarray(result)

def center_crop(img, size, in_listdataset=False):
    """
    Get the center crop of the input image
    Args:
        img: input image [HxWxC]
        size: size of the center crop (tuple) (width, height)
    Output:
        img_pad: center crop
        x, y: coordinates of the crop
    """

    if not isinstance(size, tuple):
        size = (size, size)
        #size is W,H

    img = img.copy()
    h, w = img.shape[:2]

    pad_w = 0
    pad_h = 0
    if w < size[0]:
        pad_w = np.int(np.ceil((size[0] - w) / 2))
    if h < size[1]:
        pad_h = np.int(np.ceil((size[1] - h) / 2))
    img_pad = cv2.copyMakeBorder(img,
                                 pad_h,
                                 pad_h,
                                 pad_w,
                                 pad_w,
                                 cv2.BORDER_CONSTANT,
                                 value=[0, 0, 0])
    h, w = img_pad.shape[:2]

    x1 = w // 2 - size[0] // 2
    y1 = h // 2 - size[1] // 2

    if len(img.shape)==3 :
        img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0], :]
    else: 
        img_pad = img_pad[y1:y1 + size[1], x1:x1 + size[0]]
    return img_pad, x1, y1

def get_mapping_horizontal_flipping(image):
    H, W, C = image.shape
    mapping = np.zeros((H,W,2), np.float32)
    for j in range(H):
        for i in range(W):
            mapping[j, i, 0] = W - i
            mapping[j, i, 1] = j
    return mapping, remap_using_correspondence_map(image, mapping[:,:,0], mapping[:,:,1])

def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        #torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            map = flow + grid # here also channel first
            if not output_channel_first:
                map = map.permute(1,2,0).float()
        return map.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                map[i, :, :, 0] = flow[i, :, :, 0] + X
                map[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                map = map.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            map = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            map[:,:,0] = flow[:,:,0] + X
            map[:,:,1] = flow[:,:,1] + Y
            if output_channel_first:
                map = map.transpose(2,0,1).float()
        return map.astype(np.float32)

def convert_mapping_to_flow(map, output_channel_first=True):
    if not isinstance(map, np.ndarray):
        # torch tensor
        if len(map.shape) == 4:
            if map.shape[1] != 2:
                # size is BxHxWx2
                map = map.permute(0, 3, 1, 2)

            B, C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if map.is_cuda:
                grid = grid.cuda()
            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(0,2,3,1)
        else:
            if map.shape[0] != 2:
                # size is HxWx2
                map = map.permute(2, 0, 1)

            C, H, W = map.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if map.is_cuda:
                grid = grid.cuda()

            flow = map - grid # here also channel first
            if not output_channel_first:
                flow = flow.permute(1,2,0).float()
        return flow.float()
    else:
        # here numpy arrays
        if len(map.shape) == 4:
            if map.shape[3] != 2:
                # size is Bx2xHxW
                map = map.permute(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = map.shape[:3]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                flow[i, :, :, 0] = map[i, :, :, 0] - X
                flow[i, :, :, 1] = map[i, :, :, 1] - Y
            if output_channel_first:
                flow = flow.transpose(0,3,1,2)
        else:
            if map.shape[0] == 2:
                # size is 2xHxW
                map = map.permute(1,2,0)
            # HxWx2
            h_scale, w_scale = map.shape[:2]
            flow = np.copy(map)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            flow[:,:,0] = map[:,:,0]-X
            flow[:,:,1] = map[:,:,1]-Y
            if output_channel_first:
                flow = flow.transpose(2,0,1).float()
        return flow.astype(np.float32)


def check_gt_pair(dataset):
    dataloader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=32)
    norm = []
    print("Check gt pair..")
    print("total len : ", len(dataloader))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, mini_batch in pbar:
        per_image = []
        source_img = mini_batch['source_image'].squeeze()
        target_img = mini_batch['target_image'].squeeze()
        flow = mini_batch['flow_map'].squeeze()
        mask = mini_batch['correspondence_mask'].squeeze()

        img_width = source_img.shape[2]
        img_height = source_img.shape[1]

        
        for h in range(0, img_height):
            for w in range(0, img_width):
                if(mask[h][w] == 1) : 
                    x_dt = flow[0][h][w].round().long()
                    y_dt = flow[1][h][w].round().long()
                    diff = (target_img[:,min(img_height-1, h+y_dt),min(img_width-1,w+x_dt)] - \
                            source_img[:,h,w]).double()
                    per_image.append(torch.norm(diff, p=2).numpy()) ## diff per one pixel(3 channel)
        ## mean of diff per one "IMAGE(source-target)"
        norm.append(np.asarray(per_image).mean())
    print("total norm mean : ", np.asarray(norm).mean())                


                   



        



    
    












