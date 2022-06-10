import copy
import numpy as np
from tqdm import tqdm
from skimage import img_as_float


def gaussian_wholedataset(dataset, mean, sigma):
    data = copy.deepcopy(dataset)
    for i in tqdm(range(dataset.shape[0])):
        gauss = np.random.normal(mean, sigma, data[0].shape)
        gauss = gauss.reshape(data[0].shape)
        data[i, :, :, :] = data[i, :, :, :] + gauss
    return data


def rician_wholedataset(dataset, BaseVal, Scale, NumSamples):
    data = copy.deepcopy(dataset)
    for i in tqdm(range(data.shape[0])):
        data[i, :, :, :] *= 255.0/data[i, :, :, :].max()
        noise = np.random.normal(scale=Scale, size=(
            NumSamples, 2)) + [[BaseVal, 0]]
        noise = np.linalg.norm(noise, axis=1)
        data[i, :, :, :] = img_as_float(
            data[i, :, :, :]) + noise.reshape(data[i, :, :, :].shape)
        data[i, :, :, :] *= 255.0/data[i, :, :, :].max()
        data[i] = data[i] / 255.0
    return data


# # Parameters of Rician noise
# v = 5
# s = 3
# N = 240*240  # how many samples
# rician_added = rician_wholedataset(data, v, s, N)

# much faster inplace function
def rician_wholedataset_inplace(data, BaseVal, Scale, NumSamples):
    # having it inplace helps a lot
    # having this noise thing outside outside helps w speed
    noise = np.random.normal(scale=Scale, size=(
        NumSamples, 2)) + [[BaseVal, 0]]
    noise = np.linalg.norm(noise, axis=1)
    for i in tqdm(range(data.shape[0])):
        # this has problems due to dive by zero and others
        data[i, :, :, :] *= 255.0/data[i, :, :, :].max()
        data[i, :, :, :] = img_as_float(
            data[i, :, :, :]) + noise.reshape(data[i, :, :, :].shape)
        data[i, :, :, :] *= 255.0/data[i, :, :, :].max()
        data[i] = data[i] / 255.0
    return data


def rayleigh_wholedataset(dataset, rand_scale=8, NumSamples=240):
    data = copy.deepcopy(dataset)
    for i in tqdm(range(data.shape[0])):
        raynoise = np.random.rayleigh(rand_scale, NumSamples*NumSamples)
        raynoise = raynoise.reshape(NumSamples, NumSamples)
        data[i, :, :, :] = img_as_float(
            data[i, :, :, :]*255) + raynoise.reshape(data[i, :, :, :].shape)
        data[i, :, :, :] *= 255.0/data[i, :, :, :].max()
        data[i] = data[i] / 255.0
    return data


# inplace + same noise
# deep copy costs lil but noise creation takes a lot of time
def rayleigh_wholedataset_inplace(dataset, rand_scale=8, NumSamples=240):
    raynoise = np.random.rayleigh(rand_scale, NumSamples*NumSamples)
    raynoise = raynoise.reshape(NumSamples, NumSamples)
    for i in tqdm(range(dataset.shape[0])):
        dataset[i, :, :, :] = img_as_float(
            dataset[i, :, :, :]*255) + raynoise.reshape(dataset[i, :, :, :].shape)
        dataset[i, :, :, :] *= 255.0/dataset[i, :, :, :].max()
        dataset[i] = dataset[i] / 255.0
    return dataset

# rayleigh_added = rayleigh_wholedataset(data,rand_scale,NumSamples)
