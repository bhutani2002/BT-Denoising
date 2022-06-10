import numpy as np
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import nibabel as nib
import h5py
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from skimage.transform import resize
import natsort

def load_images(PATH, imglist, num_imgs, h, w, dim):
    scaler = MinMaxScaler()
    data = np.zeros((num_imgs, h, w, dim))
    for i in tqdm(range(num_imgs)):
        img = np.asarray(Image.open(PATH + imglist[i]))
        img = scaler.fit_transform(
            img.reshape(-1, img.shape[-1])).reshape(img.shape).reshape((h, w, dim))
        data[i, :, :, :] = img
    return data



def loadfigsharedata(path, length=3064, typec="image", size=512):
    #typec needs to be careful choosen, else it will error out if wrong key is used
    imgList = os.listdir(path)
    if length != len(imgList):
        num = length
    else:
        num = len(imgList)
    data = np.zeros((num, size, size, 1))
    for i in range(0,num):
        img =  np.array(h5py.File(f"{path}/{i+1}.mat")["cjdata"][typec])
        if img.shape != (size,size):
            img = resize(img, (size,size), anti_aliasing=True)
        img = (img - np.min(img))/(np.max(img) - np.min(img))
        data[i] = img.reshape((size,size,1))
    return data

# data = loadfigsharedata("../input/figshare-brain-tumor-dataset/dataset/data", 3064, "tumorMask", 256)
# print(data.shape)


def findminmaxidx(seg, num):
    front = 0
    end = 155
    for i in range(155):
        if np.sum(seg[:,:,i]>0) > num:
            front = i
            break

    for i in range(155):
        if np.sum(seg[:,:,154-i]>0) > num:
            end = 154 - i
            break
    return range(front, end)

def singledir(path):
    
    size = 200
    imgList = natsort.natsorted(os.listdir(path))
    # auto detect the names
    flair =  nib.load(path + "/" + imgList[0]).get_fdata().astype(np.float32)
    seg = nib.load(path + "/" +imgList[1]).get_fdata().astype(np.float32)
    t1 =  nib.load(path + "/" +imgList[2]).get_fdata().astype(np.float32)
    t1ce =  nib.load(path + "/" + imgList[3]).get_fdata().astype(np.float32)
    t2 =  nib.load(path + "/" +imgList[4]).get_fdata().astype(np.float32)
    res = findminmaxidx(seg, size)
    
    
    flair = np.rollaxis(flair, 2,0).reshape(155,240, 240, 1)[res]
    seg = np.rollaxis(seg, 2,0).reshape(155,240, 240, 1)[res]
    t1 = np.rollaxis(t1, 2,0).reshape(155,240, 240, 1)[res]
    t1ce = np.rollaxis(t1ce, 2,0).reshape(155,240, 240, 1)[res]
    t2 = np.rollaxis(t2, 2,0).reshape(155,240, 240, 1)[res]
    return  t1, t1ce, flair, t2, seg

def createdataset(PATH, OUTPUT_PATH, numpatients=1, npzmode=True):
    dirList = natsort.natsorted(os.listdir(PATH))[0:numpatients]
    dirPaths = [os.path.join(PATH + i) for i in dirList]
    
    IMG_WIDTH = 240
    IMG_HEIGHT = 240 
    DIMS = 1
    
    t1 = np.zeros((0,IMG_WIDTH, IMG_HEIGHT, DIMS))
    t1ce = np.zeros((0,IMG_WIDTH, IMG_HEIGHT, DIMS))
    flair = np.zeros((0,IMG_WIDTH, IMG_HEIGHT, DIMS))
    t2= np.zeros((0,IMG_WIDTH, IMG_HEIGHT, DIMS))
    seg = np.zeros((0,IMG_WIDTH, IMG_HEIGHT, DIMS))
    
    try:
        # Setup multiprocessing pool
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes = cpu_count)  
        for data  in pool.map(singledir, dirPaths):
            t1 = np.concatenate((t1,data[0]))
            t1ce = np.concatenate((t1ce,data[1]))
            flair = np.concatenate((flair,data[2]))
            t2 = np.concatenate((t2,data[3]))
            seg = np.concatenate((seg,data[4]))

    finally:
        pool.close()
        pool.join()

    # make sure to create a directory if it doesn't exist, errors out here
    if npzmode == True:
        np.savez(OUTPUT_PATH + "t1.npz", t1)
        np.savez(OUTPUT_PATH + "t1ce.npz", t1ce)
        np.savez(OUTPUT_PATH + "flair.npz", flair)
        np.savez(OUTPUT_PATH + "t2.npz", t2)
        np.savez(OUTPUT_PATH + "seg.npz", seg)
        return 0 
    else:
        return {"t1": t1,"t1ce": t1ce,"flair": flair, "t2": t2, "seg": seg}
    
    return -1
    
# PATH = "./test_files/dataset/"
# OUTPUT_PATH = "./test_files/output/"
# d = createdataset(PATH, OUTPUT_PATH, 2, True)