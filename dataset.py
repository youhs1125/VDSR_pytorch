import os
from PIL import Image
import numpy as np
import cv2
import torch
import torch.utils.data as td
import matplotlib.pyplot as plt

def getImageFiles(path):
    img_path = []

    for p in path:
        for (root, directories, files) in os.walk(p):
            for file in files:
                file_path = os.path.join(root,file)
                img_path.append(file_path)

    file = []
    for i in range(len(img_path)):
        img = Image.open(img_path[i])
        img = np.array(img, dtype=np.float32)
        img /= 255.
        file.append(img)

    print("images: ",len(file))
    return file

def DataAugmentation(file):
    len_file = len(file)
    for i in range(len_file):
        file.append(cv2.rotate(file[i],cv2.ROTATE_90_CLOCKWISE))
        file.append(cv2.rotate(file[i],cv2.ROTATE_180))
        file.append(cv2.rotate(file[i],cv2.ROTATE_90_COUNTERCLOCKWISE))

    len_file = len(file)
    for i in range(len_file):
    # flip Left-Right
        file.append(cv2.flip(file[i],1))
    # flip Top-Bottom
        tempimg = cv2.flip(file[i],0)
        file.append(tempimg)
    # flip Left-Right + Top-Bottom
        file.append(cv2.flip(tempimg,1))

    print("Augmentation Finished size:",len(file))
    return file

def getSubImages(file, imgsize):
    size = imgsize
    for i in range(len(file)):
        size = min(file[i].shape[0],file[i].shape[1],size)
    print("subImages size: ",size,"x",size)

    target = []

    for i in range(len(file)):
        img_h = (file[i].shape[0] - size) // 2
        img_w = (file[i].shape[1] - size) // 2
        sample = file[i][img_h:img_h + size, img_w:img_w + size]
        target.append(sample)
    target = np.array(target)

    print(target[0].shape)
    return target

def downsampling(file, isTest=False):
    ds = []
    for i in range(len(file)):
        img_W,img_H = file[i].shape[0], file[i].shape[1]
        temp = cv2.GaussianBlur(file[i], (0, 0), 1)
        temp = cv2.resize(temp, dsize=(img_H//2, img_W//2), interpolation=cv2.INTER_CUBIC)
        temp = cv2.resize(temp, dsize=(img_H, img_W), interpolation=cv2.INTER_CUBIC)
        ds.append(temp)

    if isTest == False:
        ds = np.array(ds)
    return ds

def changeColorChannelLocation(file1, file2):
    data = np.ascontiguousarray(file1.transpose((0,3,1,2)))
    target = np.ascontiguousarray(file2.transpose((0,3,1,2)))

    return data, target


def getDataset():
    path = []

    path.append("Images/BSDS200")
    path.append("Images/T91")

    file = getImageFiles(path)
    file = DataAugmentation(file)
    tfile = getSubImages(file, 41)
    dfile = downsampling(tfile)

    target, data = changeColorChannelLocation(tfile,dfile)

    # define dataset
    target = torch.from_numpy(target)
    data = torch.from_numpy(data)

    dataset = td.TensorDataset(data, target)

    # split train, validation

    train_val_ratio = 0.8

    train_size = int(data.shape[0] * train_val_ratio)
    val_size = data.shape[0] - train_size

    train_data, val_data = td.random_split(dataset, [train_size, val_size])

    # define dataloader

    train_dataloader = td.DataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = td.DataLoader(val_data, batch_size=64, shuffle=False)
    print(len(train_dataloader), len(val_dataloader))

    return train_dataloader, val_dataloader


def getTestData():
    path = []

    # path.append("Images/BSDS100")
    path.append("Images/Set14/original")
    path.append("Images/Set5/original")
    # path.append("Images/urban100")

    data = getImageFiles(path)
    target = downsampling(data,isTest=True)

    return data, target