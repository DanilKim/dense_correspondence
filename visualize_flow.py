import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import pdb

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


if __name__ == "__main__":
    path = os.path.join('sintel', 'extracted_pairs', 'clean', 'mag_20_0.2', 'bandage_2', 'start_26', 'end_44')
    image_1 = imread(os.path.join(path, 'source.png'))
    image_2 = imread(os.path.join(path, 'target.png'))
    flow    = load_flo(os.path.join(path, 'flow.flo'))
    mask    = imread(os.path.join(path, 'occlusion.png'))
    mask    = 1 - mask
    valid_points_y, valid_points_x = np.nonzero(mask)

    height, width = image_1.shape[:2]

    margin = 30
    composite = np.zeros((2 * height + margin, width, 3),) + 255
    v_off = height + margin
    composite[:height, :width, :] = image_1 / 255
    composite[v_off:v_off+height, :width, :] = image_2 / 255

    X, Y = np.meshgrid(np.linspace(0, width-1, width),
                       np.linspace(0, height-1, height))
    map_x = (X + flow[:,:,0]).astype(np.float32)
    map_y = (Y + flow[:,:,1]).astype(np.float32)
    remapped_image_1 = cv2.remap(image_1, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    remapped_image_2 = cv2.remap(image_2, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    masked_remapped_image_2 = remapped_image_2 * (mask[:,:,np.newaxis]/255)

    image_1 = torch.from_numpy(image_1)
    image_2 = torch.from_numpy(image_2)
    plt.figure(figsize=(18, 9))

    plt.subplot(321)
    plt.imshow(image_1)
    #plt.set_title("image_1")

    plt.subplot(323)
    plt.imshow(image_2)
    #plt.set_title("image_2")

    plt.subplot(325)
    plt.imshow(remapped_image_2)
    #plt.set_title("remapped_image_2")

    plt.subplot(122)
    plt.imshow(composite)
    cnt = 0
    AEPE = 0
    for x, y in zip(valid_points_x, valid_points_y):
        cnt += 1
        pix_1 = image_1[y,x,:]
        pix_2 = image_2[int(map_y[y,x]), int(map_x[y,x]), :]
        AEPE += np.linalg.norm(pix_1-pix_2)
        if cnt % 10000 != 1:
            continue
        plt.plot([x, map_x[y,x]], [y, v_off + map_y[y,x]], 'r', linewidth=0.5)
        print('---------------')
        print(pix_1)
        print(pix_2)
        print(np.subtract(pix_1,pix_2))
        print(np.linalg.norm(pix_1-pix_2))
        #print(x, y, int(map_x[y,x]), int(map_y(y,x)), float(pix_1[0]), float(pix_1[1]), float(pix_1[2]), float(pix_2[0]), float(pix_2[1]), float(pix_2[2]))
        #print("(x,y) = (%d,%d), (x',y') = (%d,%d) / I_s(x,y) = (%.2f,%.2f,%.2f), I_t(x',y') = (%.2f,%.2f,%.2f)"
        #      % (x, y, int(map_x[y,x]), int(map_y(y,x)), float(pix_1[0]), float(pix_1[1]), float(pix_1[2]), float(pix_2[0]), float(pix_2[1]), float(pix_2[2])))
    #plt.set_title("correspondence")
    #print("AEPE: %.3f" % float(AEPE)/cnt)
    print(AEPE, cnt, AEPE/cnt)
    plt.show()

    #fig, ax = plt.subplots(4,1,figsize=(15,15))
    #ax[0].imshow(image_1)
    #ax[0].set_title("image 1")
    #ax[1].imshow(image_2)
    #ax[1].set_title("image 2")
    ##ax[2].imshow(remapped_image_1)
    ##ax[2].set_title("remapped image 1")
    #ax[2].imshow(remapped_image_2)
    #ax[2].set_title("remapped image 2")
    ##ax[4].imshow(masked_remapped_image_2)
    ##ax[4].set_title("masked remapped image 2")
    #ax[3].imshow(composite)
    #cnt = 0
    #for x, y in zip(valid_points_x, valid_points_y):
    #    cnt += 1
    #    if cnt % 5000 != 1:
    #        continue
    #    ax[3].plot([x, map_x[y,x]], [y, v_off + map_y[y,x]], 'r', linewidth=0.5)
    #ax[3].set_title("correspondence")
    #fig.show()
