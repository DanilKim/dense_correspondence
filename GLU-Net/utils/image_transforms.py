import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from datasets.util import random_crop, center_crop, resize


def TensorToArray(tensor, type):
    """Converts a torch.FloatTensor of shape (C x H x W) to a numpy.ndarray (H x W x C) """
    array=tensor.cpu().detach().numpy()
    if len(array.shape)==4:
        if array.shape[3] > array.shape[1]:
            # shape is BxCxHxW
            array = np.transpose(array, (0,2,3,1))
    else:
        if array.shape[2] > array.shape[0]:
            # shape is CxHxW
            array=np.transpose(array, (1,2,0))
    return array.astype(type)


# class ToTensor of torchvision also normalised to 0 1
class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, get_float=True):
        self.get_float=get_float

    def __call__(self, array):

        if not isinstance(array, np.ndarray):
            array = np.array(array)
        if len(array.shape) == 3:
            array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        if self.get_float:
            return tensor.float()
        else:
            return tensor


class RandomCrop(object):
    """Randomly crop input numpy image array (H x W x C) to (crop_size x crop_size x C).
       crop_size <= min(H, W)"""
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.seed = 2020
        self.value = [0,0,0]

    def __call__(self, img):
        return random_crop(img, self.crop_size, self.seed, self.value)


class CenterCrop(object):
    """Center crop input numpy image array (H x W x C) to (crop_size x crop_size x C).
       crop_size <= min(H, W)"""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        return center_crop(img, self.crop_size)[0]


class Resize(object):
    """Resize input numpy image array (H x W x C) to (size x size x C).
       size <= min(H, W)"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return resize(img, self.size)


class Raw(object):
    """ Identity transform (input image = output image)"""
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return img


class ResizeFlow(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, size):
        if not isinstance(size, tuple):
            size = (size, size)
        self.size = size

    def __call__(self, tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _, h_original, w_original = tensor.shape
        resized_tensor = F.interpolate(tensor.unsqueeze(0), self.size, mode='bilinear', align_corners=False)
        resized_tensor[:,0,:,:] *= float(self.size[1])/float(w_original)
        resized_tensor[:,1,:,:] *= float(self.size[0])/float(h_original)
        return resized_tensor.squeeze(0)


class RGBtoBGR(object):
    """converts the RGB channels of a numpy array HxWxC into RGB"""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        ch_arr = [2, 1, 0]
        img = array[..., ch_arr]
        return img

