import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
from datasets.util import load_flo
import torch
from datasets.util import random_crop, resize, center_crop
import torchvision.transforms as transforms

def get_gt_correspondence_mask(flow):
    # convert flow to mapping
    h,w = flow.shape[:2]
    X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                       np.linspace(0, h - 1, h))
    map_x = (flow[:,:,0]+X).astype(np.float32)
    map_y = (flow[:,:,1]+Y).astype(np.float32)

    mask_x = np.logical_and(map_x>0, map_x< w)
    mask_y = np.logical_and(map_y>0, map_y< h)
    mask = np.logical_and(mask_x, mask_y).astype(np.uint8)
    return mask


def image_flow_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root,path) for path in path_imgs]
    flo = os.path.join(root,path_flo)
    return [imread(img).astype(np.uint8) for img in imgs], load_flo(flo)


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, source_image_transform=None, target_image_transform=None, flow_transform=None,
                 co_transform=None, loader=image_flow_loader, mask=False, size=False):
        """

        :param root: directory containing the dataset images
        :param path_list: list containing the name of images and corresponding ground-truth flow files
        :param source_image_transform: transforms to apply to source images
        :param target_image_transform: transforms to apply to target images
        :param flow_transform: transforms to apply to flow field
        :param co_transform: transforms to apply to both images and the flow field
        :param loader: loader function for the images and the flow
        :param mask: bool indicating is a mask of valid pixels needs to be loaded as well from root
        :param size: size of the original source image
        outputs:
            - source_image
            - target_image
            - flow_map
            - correspondence_mask
            - source_image_size
        """

        self.root = root
        self.path_list = path_list
        self.source_image_transform = source_image_transform
        self.target_image_transform = target_image_transform
        self.flow_transform = flow_transform
        self.co_transform = co_transform
        self.loader = loader
        self.mask = mask
        self.size = size
    
    def get_path(self):
        return self.path_list

    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_flow = self.path_list[index]

        if not self.mask:
            if self.size:
                inputs, gt_flow, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                inputs, gt_flow = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
            mask = self.source_image_transform(mask).long()
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)

        W_source = inputs[0].shape[2]
        H_source = inputs[0].shape[1]

        source = inputs[0][:, :int(H_source / 16) * 16, :int(W_source / 16) * 16]

        W_target = inputs[1].shape[2]
        H_target = inputs[1].shape[1]

        target = inputs[1][:, :int(H_target / 16) * 16, :int(W_target / 16) * 16]

        W_flow = gt_flow.shape[2]
        H_flow = gt_flow.shape[1]

        flow = gt_flow[:, :int(H_flow / 16) * 16, :int(W_flow / 16) * 16]

        W_mask = mask.shape[1]
        H_mask = mask.shape[0]

        mask_trans = mask[:int(H_mask / 16) * 16, :int(W_mask / 16) * 16]

        return {'source_image': source,
                'target_image': target,
                'flow_map': flow,
                'correspondence_mask': mask_trans,
                'source_image_size': source_size
                }

    def __len__(self):
        return len(self.path_list)


class SintelAllpairListDataset(ListDataset):
    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_flow, occ_mask = self.path_list[index]
        if not self.mask:
            if self.size:
                inputs, gt_flow, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                inputs, gt_flow = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow, occ_mask)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow, occ_mask)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)
        # here gt_flow has shape HxWx2
        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.source_image_transform is not None:
            if self.transform_type == 'raw':
                source_trans = self.source_image_transform(inputs[0])
            elif self.transform_type == 'random':
                source_trans = random_crop(inputs[0], self.crop_size, self.fixed_np_seed)
                source_trans = self.source_image_transform(source_trans)
            elif self.transform_type == 'center': 
                source_trans, _, _ = center_crop(inputs[0], self.crop_size)
                source_trans = self.source_image_transform(source_trans)
            else : raise "transform type ERROR in listdataset.py!"

        if self.target_image_transform is not None:
            if self.transform_type == 'raw':
                target_trans = self.target_image_transform(inputs[1])
            elif self.transform_type == 'random':
                target_trans = random_crop(inputs[1],self.crop_size, self.fixed_np_seed)
                target_trans = self.target_image_transform(target_trans)
            elif self.transform_type == 'center': 
                target_trans, _, _ = center_crop(inputs[1], self.crop_size)
                target_trans = self.target_image_transform(target_trans)
            else : raise "transform type ERROR in listdataset.py!"

        if self.flow_transform is not None:
            if self.transform_type == 'raw':
                gt_flow_trans = self.flow_transform(gt_flow)
            elif self.transform_type == 'random':
                gt_flow_trans = random_crop(gt_flow,self.crop_size, self.fixed_np_seed)
                gt_flow_trans = self.flow_transform(gt_flow_trans)
            elif self.transform_type == 'center': 
                gt_flow_trans, _, _ = center_crop(gt_flow, self.crop_size)
                gt_flow_trans = self.flow_transform(gt_flow_trans)
            else : raise "transform type ERROR in listdataset.py!"

        if mask is not None :
            if self.transform_type == 'raw':
                mask_trans = torch.from_numpy(mask).long()
            elif self.transform_type == 'random':
                mask_trans = random_crop(mask, self.crop_size, self.fixed_np_seed, value=[0]) ## invaldate pixel for 0
                mask_trans = torch.from_numpy(mask_trans).long()
            elif self.transform_type == 'center':
                #mask_trans = torch.from_numpy(misc.imresize(mask, (self.crop_size, self.crop_size), 'bilinear')).long()
                resize_trans, _, _ = center_crop(mask, self.crop_size)
                mask_trans = torch.from_numpy(resize_trans).long()
            else : raise "transform type ERROR in listdataset.py!"

        return {'source_image': source_trans,
                'target_image': target_trans,
                'flow_map': gt_flow_trans,
                'correspondence_mask': mask_trans,
                'source_image_size': source_size
                }

class KittiListDataset(ListDataset):
    def __getitem__(self, index):
        # for all inputs[0] must be the source and inputs[1] must be the target
        inputs, gt_flow = self.path_list[index]
        if not self.mask:
            if self.size:
                inputs, gt_flow, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                inputs, gt_flow = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            if self.co_transform is not None:
                inputs, gt_flow = self.co_transform(inputs, gt_flow)

            mask = get_gt_correspondence_mask(gt_flow)
        else:
            if self.size:
                inputs, gt_flow, mask, source_size = self.loader(self.root, inputs, gt_flow)
            else:
                # loader comes with a mask of valid correspondences
                inputs, gt_flow, mask = self.loader(self.root, inputs, gt_flow)
                source_size = inputs[0].shape
            # mask is shape hxw
            if self.co_transform is not None:
                inputs, gt_flow, mask = self.co_transform(inputs, gt_flow, mask)

        # here gt_flow has shape HxWx2

        # after co transform that could be reshapping the target
        # transforms here will always contain conversion to tensor (then channel is before)
        if self.source_image_transform is not None:
            inputs[0] = self.source_image_transform(inputs[0])
        if self.target_image_transform is not None:
            inputs[1] = self.target_image_transform(inputs[1])
        if self.flow_transform is not None:
            gt_flow = self.flow_transform(gt_flow)
        mask = mask.astype(np.uint8)

        W_source = inputs[0].shape[2]
        H_source = inputs[0].shape[1]

        source = inputs[0][:, :int(H_source/16)*16, :int(W_source/16)*16]

        W_target = inputs[1].shape[2]
        H_target = inputs[1].shape[1]
        
        target = inputs[1][:, :int(H_target/16)*16, :int(W_target/16)*16]

        W_flow = gt_flow.shape[2]
        H_flow = gt_flow.shape[1]

        flow = gt_flow[:, :int(H_flow/16)*16, :int(W_flow/16)*16]

        W_mask = mask.shape[1]
        H_mask = mask.shape[0]

        mask_trans = mask[ :int(H_mask/16)*16, :int(W_mask/16)*16]
        mask_trans = torch.from_numpy(mask_trans).long()

        return {'source_image': source ,
                'target_image': target,
                'flow_map': flow ,
                'correspondence_mask': mask_trans,
                'source_image_size': source.shape
                }
    
