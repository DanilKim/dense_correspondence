import os.path
import glob
from .listdataset import ListDataset
from datasets.util import split2list, collate_fn_make_same_size
from datasets.dataset_split import train_test_split_dir
import torchvision.transforms as transforms
from utils.image_transforms import ArrayToTensor
from datasets.mpisintel import mpi_sintel_allpair
from datasets.KITTI_optical_flow import KITTI_occ as kitti_occ
from torch.utils.data import DataLoader


def make_dataset(dir, split=None, dataset_name=None):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm  in folder images and
      [name]_flow.flo' in folder flow '''
    images = []
    '''
    if get_mapping:
        flow_dir = 'mapping'
        # flow_dir is actually mapping dir in that case, it is always normalised to [-1,1]
    '''
    flow_dir = 'flow'
    image_dir = 'images'
    mask_dir = 'mask'

    # Make sure that the folders exist
    if not os.path.isdir(dir):
        raise ValueError("the training directory path that you indicated does not exist ! ")
    if not os.path.isdir(os.path.join(dir, flow_dir)):
        raise ValueError("the training directory path that you indicated does not contain the flow folder ! Check your directories.")
    if not os.path.isdir(os.path.join(dir, image_dir)):
        raise ValueError("the training directory path that you indicated does not contain the images folder ! Check your directories.")

    for flow_map in sorted(glob.glob(os.path.join(dir, flow_dir, '*_flow.flo'))):
        flow_map = os.path.join(flow_dir, os.path.basename(flow_map))
        root_filename = os.path.basename(flow_map)[:-9]
        img1 = os.path.join(image_dir, root_filename + '_img_1.jpg') # source image
        img2 = os.path.join(image_dir, root_filename + '_img_2.jpg') # target image
        occ_mask = os.path.join(mask_dir, root_filename + '_occlusion.png')
        if not (os.path.isfile(os.path.join(dir, img1)) and os.path.isfile(os.path.join(dir, img2))):
            continue
        if dataset_name is not None:
            img1 = os.path.join(dataset_name, img1),
            img2 = os.path.join(dataset_name, img2),
            flow_map = os.path.join(dataset_name, flow_map)
            occ_mask = os.path.join(dataset_name, occ_mask)

        images.append([[img1, img2], flow_map, occ_mask])
    return split2list(images, split, default_split=0.97)

def PreMadeDataset(root, source_image_transform=None, target_image_transform=None, flow_transform=None,
                   co_transform=None, split=None, mask=True, transform_type='raw', crop_size=512):
    # that is only reading and loading the data and applying transformations to both datasets

    if isinstance(root, list):
        train_list=[]
        test_list=[]
        for sub_root in root:
            _, dataset_name = os.path.split(sub_root)
            print(dataset_name)
            sub_train_list, sub_test_list = make_dataset(sub_root, split, dataset_name=dataset_name)
            train_list.extend(sub_train_list)
            test_list.extend(sub_test_list)
        root = os.path.dirname(sub_root)
    else:
        train_list, test_list = make_dataset(root, split)
    print(root)
    train_dataset = ListDataset(root, train_list, source_image_transform=source_image_transform,
                                target_image_transform=target_image_transform, mask=mask,
                                flow_transform=flow_transform, co_transform=co_transform)
    test_dataset = ListDataset(root, test_list, source_image_transform=source_image_transform,
                               target_image_transform=target_image_transform, mask=mask,
                               flow_transform=flow_transform, co_transform=co_transform)

    return train_dataset, test_dataset


def CombinedDataset(dataset_path, dataset_list, batch_size= 10, num_workers=10, is_specific=True, dataset_name=None):
    #####################################
    # Available Dataset List 
    candidate_list = ['sintel_allpair', 'kitti_2012', 'kitti_2015']

    #####################################
    # Default Transforms
    source_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform = transforms.Compose([ArrayToTensor()])

    ## for sintel_allpair
    source_img_transforms_sintel = transforms.Compose([ArrayToTensor(get_float=False)])
    target_img_transforms_sintel = transforms.Compose([ArrayToTensor(get_float=False)])
    flow_transform_sintel = transforms.Compose([ArrayToTensor()])

    #####################################
    # Write Down Data-specific Dataloader(return form must be 'Tuple')
    sintel_allpair = mpi_sintel_allpair(root = dataset_path, dataset="sintel_allpair", 
                                        source_image_transform=source_img_transforms_sintel,
                                        target_image_transform=target_img_transforms_sintel, 
                                        flow_transform=flow_transform_sintel, 
                                        co_transform=None, split=0.7, 
                                        mask=True, transform_type='raw', crop_size=512)

    kitti_2012 = kitti_occ(root = os.path.join(dataset_path, 'KITTI_2012/training/'), 
                                source_image_transform=source_img_transforms,
                                target_image_transform=target_img_transforms, 
                                flow_transform=flow_transform)
    kitti_2015 = kitti_occ(root = os.path.join(dataset_path, 'KITTI_2015/training/'), 
                                source_image_transform=source_img_transforms,
                                target_image_transform=target_img_transforms, 
                                flow_transform=flow_transform)
    
    ######################################
    # Available dataloader List in dictionary
    candidate_loader_list = [
                            {'kitti_2012' : kitti_2012},
                            {'kitti_2015' : kitti_2015},
                            {'sintel_allpair' : sintel_allpair} 
                            ]


    if not is_specific : # training all dataset with 'Listdataset'
        data_list = []
        for sub_dataset in dataset_list:
            print(sub_dataset + " loaded.")
            for candidate in candidate_list:
                if(sub_dataset in candidate or candidate in sub_dataset) : 
                    for loader in candidate_loader_list:
                        if(candidate == sub_dataset and candidate in loader):
                            train_dataloader = DataLoader(loader[candidate][0],
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           num_workers=num_workers,
                                                           collate_fn = collate_fn_make_same_size)

                            val_dataloader = DataLoader(loader[candidate][1],
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        num_workers=num_workers,
                                                        collate_fn = collate_fn_make_same_size
                                                        )
                            data_list.append((train_dataloader, val_dataloader))

        return data_list
        

    elif is_specific: # training specific dataset with its own dataset function.
        for candidate in candidate_list:
            if(dataset_name in candidate or candidate in dataset_name) : 
                for loader in candidate_loader_list:
                    if(candidate in loader):
                        train_dataloader = DataLoader(loader[candidate][0],
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=num_workers,
                                                      collate_fn = collate_fn_make_same_size
                        )
                        val_dataloader = DataLoader(loader[candidate][1],
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    collate_fn = collate_fn_make_same_size
                        )

                        return [(train_dataloader, val_dataloader)]

    raise "Error!"

