#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import scipy
import torch
import numpy as np
import flow_transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.nn.functional import grid_sample
from imageio import imread
import random
import imageio


# In[2]:


class ExtractDataset:
    def __init__(self, root, scene_dir_list, data_type, save_dir="./save"):
        self.root = root
        self.type = data_type
        self.scene_dir_list = scene_dir_list
        self.data = self.get_data(scene_dir_list)
        self.save_dir = save_dir
        
        
    def load_flo(self, path):
        with open(path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
            h = np.fromfile(f, np.int32, count=1)[0]
            w = np.fromfile(f, np.int32, count=1)[0]
            data = np.fromfile(f, np.float32, count=2*w*h)
        # Reshape data into 3D array (columns, rows, bands)
        data2D = np.resize(data, (w, h, 2))
        return data2D
    
    
    def default_loader(self, root, path_img, path_flo, path_occ):
        #imgs = [os.path.join(root,path) for path in path_imgs]
        img = os.path.join(root,path_img)
        if path_flo is not None:
            flo = self.load_flo(os.path.join(root,path_flo))
            occ = imread(os.path.join(root,path_occ))
        else:
            flo = None; occ = None;
        return imread(img).astype(np.float32), flo, occ
        #return [imread(img).astype(np.float32) for img in imgs], self.load_flo(flo), imread(occ)
        

    def save_flo(self, filename, flow):
        TAG_STRING = b'PIEH'
        # torch.Size([436, 1024, 2])
        height, width, nBands = np.shape(flow)
        
        u = flow[: , : , 0]
        v = flow[: , : , 1]
        
        height, width = u.shape
        f = open(filename,'wb')
        f.write(TAG_STRING)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        tmp = np.zeros((height, width*nBands))
        tmp[:,np.arange(width)*2] = u
        tmp[:,np.arange(width)*2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        f.close()
        
    
    def get_data(self, scene_dir_list):
        whole_file = []
        for scene_dir in self.scene_dir_list:
            single_dir_file = []
            filelist = sorted(glob.glob(os.path.join(self.root,'flow',scene_dir,'*.flo')))
            for flow_map in filelist:
                flow_map = os.path.relpath(flow_map, os.path.join(self.root,'flow'))
                
                scene_dir, filename = os.path.split(flow_map)
                no_ext_filename     = os.path.splitext(filename)[0]
                prefix,    frame_nb = no_ext_filename.split('_')
                frame_nb = int(frame_nb)
                
                occ_mask = os.path.join('occlusions', scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
                flow_map = os.path.join('flow', flow_map)
                
                img = os.path.join('clean', scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
                if (os.path.isfile(os.path.join(self.root, img))):
                    single_dir_file.append([img, flow_map, occ_mask])
            
            ## Add the last frame
            img = os.path.join('clean', scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb+1))
            if (os.path.isfile(os.path.join(self.root, img))):
                single_dir_file.append([img, None, None])
            
            whole_file.append(single_dir_file)
            
        return whole_file
    
    
    def extract_data(self):
        if not(os.path.isdir(self.save_dir)):
            os.makedirs(os.path.join(self.save_dir))
            
        for dir_num, dir_list in enumerate(self.data): # per scene directory
            
            scene_name = self.scene_dir_list[dir_num]
            scene_data = self.data[dir_num]
            
            #### start_frame  =  cnt + 1,  (cnt \in [0, number_of_frames-5])
            #### end_frames   =  randomly choose 'how_many_pick \in [2, number_of_frames - start_frame - 2]'
            ####                 numbers from [start_frame + 2, number_of_frames]
            
            for cnt, file in enumerate(dir_list): # per start_frame in a scene 
                if((cnt + 4) == len(dir_list)) :
                    break
                how_many_pick = random.randint(2, min(len(dir_list)-cnt-3, int(len(dir_list)/6)))
                selected_ends = self.get_random_num(cnt+1, len(dir_list), how_many_pick)
                start_num = cnt + 1      # int(file[0].split('/')[2].split('.')[0].split('_')[1])
                        
                for end_idx, end_num in enumerate(selected_ends): 
                    start_img, end_img, flow_map, survived_mask =                                 self.get_flo(int(start_num), int(end_num), scene_data)
                    occ_mask = (1 - survived_mask) * 255
                    
                    path = os.path.join(self.save_dir, scene_name, 'start_'+str(start_num), 'end_'+str(end_num))
                    if not(os.path.isdir(path)):
                        os.makedirs(path)
                        
                    imageio.imwrite(path+'/source.png', start_img.astype(np.uint8))
                    imageio.imwrite(path+'/target.png', end_img.astype(np.uint8))
                    imageio.imwrite(path+'/occlusion.png', occ_mask.numpy().astype(np.uint8))
                    self.save_flo(path+'/flow.flo', flow_map)
                    
                    
    def get_flo(self, start_frame, end_frame, train_samples):
        inputs, target, mask = train_samples[start_frame-1]
        inputs, target, mask = self.default_loader(self.root, inputs, target, mask)
        height, width, _ = target.shape
        
        ## Grid Location Point (x,y) matrix of size H X W
        Y, X = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        Grid = torch.stack((X, Y), 2).float()
        
        
        ## Survived points in start frame image.  H x W
        survived_mask = torch.ones(height, width)    

        ## new_Grid as transformed pixel location, according to each pixel location in source frame
        new_Grid = Grid.clone()
        new_Grid_norm = torch.zeros(Grid.size()).unsqueeze(0)

        for ind in range(start_frame, end_frame):
            imgs, target, occ_mask = train_samples[ind-1]
            target = torch.from_numpy(self.load_flo(os.path.join(self.root, target)))   ## Flow annotation. H x W x 2
            occ_mask = torch.from_numpy(imread(os.path.join(self.root, occ_mask))).float() / 255    ## Occlusion mask. H x W
    
            if ind == start_frame:
                warped_flow = target.permute(2,0,1)
                warped_occ_mask = occ_mask
            else:
                ## Warping Occlusion Mask
                warped_flow     = grid_sample(target.permute(2,0,1).unsqueeze(0),
                                      new_Grid_norm, mode='nearest',
                                      align_corners=True)[0]
                ## Warping Flow information
                warped_occ_mask = grid_sample(occ_mask[(None,)*2 + (...,)], 
                                      new_Grid_norm, mode='nearest', 
                                      align_corners=True)[0][0]

            survived_mask = survived_mask * (1 - warped_occ_mask)
    
            new_Grid[:,:,0] = torch.clamp((new_Grid[:,:,0] + warped_flow[0,:,:]), 0, width-1)
            new_Grid[:,:,1] = torch.clamp((new_Grid[:,:,1] + warped_flow[1,:,:]), 0, height-1)
    
            new_Grid_norm[:,:,:,0] = 2.0*new_Grid[:,:,0].clone() / max(width-1,1)-1.0       
            new_Grid_norm[:,:,:,1] = 2.0*new_Grid[:,:,1].clone() / max(height-1,1)-1.0
    
        #### Get Dense Flow Field ####
        flow = new_Grid - Grid

        inputs, target, mask = train_samples[start_frame-1]
        inputs, _, _ = self.default_loader(self.root, inputs, target, mask)
        outputs, target, occ_mask = train_samples[end_frame-1]    
        outputs, _, _ = self.default_loader(self.root, outputs, target, occ_mask)
        
        #if start_frame == 15:
        #    self.visualize(inputs, outputs, start_frame, end_frame, new_Grid_norm, survived_mask)
        return inputs[2:-2,:,:], outputs[2:-2,:,:], flow[2:-2,:,:], survived_mask[2:-2,:]
   
                
    def get_random_num(self, start, finish, num):
        if(start >= finish) : print('finish number is same or less than start num');return -1
        if(num <= 1) : print('The number of data requires bigger then one'); return -1
        return sorted(random.sample(range(start+2, finish+1), num))
        
        
    def visualize(self, src, tar, start_frame, end_frame, new_Grid_norm, survived):
        height, width, _ = tar.shape
        src = torch.from_numpy(src).permute(2,0,1).unsqueeze(0)
        tar = torch.from_numpy(tar).permute(2,0,1).unsqueeze(0)
        
        warped_target = grid_sample(tar, new_Grid_norm, mode='nearest', align_corners=True)
        warped_target = warped_target * survived + 255 * (1 - survived) 
        
        fig = plt.figure(figsize=(15,15))
        ax1 = fig.add_subplot(3,1,1)
        ax1.set_title('Frame <%d>'% start_frame, fontsize=20)
        ax1.axis("off")
        ax1.imshow(src[0].permute(1,2,0).numpy()/255)
        ax2 = fig.add_subplot(3,1,2)
        ax2.set_title('Frame <%d>'% end_frame, fontsize=20)
        ax2.axis("off")
        ax2.imshow(tar[0].permute(1,2,0).numpy()/255)
        ax3 = fig.add_subplot(3,1,3)
        ax3.set_title('Warped From <%d> to <%d>'% (end_frame, start_frame), fontsize=20)
        ax3.axis("off")
        ax3.imshow(warped_target[0].permute(1,2,0).numpy()/255)

        


# In[3]:

print("start")
Test = ExtractDataset('../sintel', ['alley_1', 'ambush_4', 'temple_2'], 'clean')
#Test = ExtractDataset('sintel', ['alley_1'], 'clean')


# In[4]:


Test.extract_data()


# In[ ]:




