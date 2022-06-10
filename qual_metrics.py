from tqdm import tqdm
import numpy as np
import multiprocessing
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from itertools import repeat

def psnr_wholedataset(dataset_original, dataset_denoised):

    avgpsnr = 0
    for i in tqdm(range(dataset_original.shape[0])):
        true_min, true_max = np.min(dataset_original[i, :, :, :]), np.max(dataset_original[i, :, :, :])
        dataRange = abs(true_min)+abs(true_max)
        psnr = peak_signal_noise_ratio(
            dataset_original[i], dataset_denoised[i], data_range=dataRange)
        avgpsnr += psnr
        avgpsnr = avgpsnr/2         

    return avgpsnr


# psnr_wholedataset(data, rayleigh_added)



def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)
def starmap_with_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)

def psnr_wholedataset_pool(dataset_original, dataset_denoised):
    true_min, true_max = np.min(dataset_original[i, :, :, :]), np.max(dataset_original[i, :, :, :])
    dataRange = abs(true_min)+abs(true_max)
    # args_iter = zip(list(unstack(dataset_original,0)), list(unstack(dataset_denoised,0)))
    args_iter = zip(list(unstack(dataset_original,0)), list(unstack(dataset_denoised,0)))   
    try:
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes = cpu_count) 
        kwargs_iter = repeat(dict(data_range=dataRange))
        branches = starmap_with_kwargs(pool, peak_signal_noise_ratio , args_iter, kwargs_iter)
    finally:
        pool.close()
        pool.join()

    return sum(branches)/dataset_original.shape[0]


def psnr_wholedataset_pool1(data):
    true_min, true_max = np.min(dataset_original[i, :, :, :]), np.max(dataset_original[i, :, :, :])
    dataRange = abs(true_min)+abs(true_max)
    args_iter = data 
    try:
        cpu_count = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes = cpu_count) 
        kwargs_iter = repeat(dict(data_range=dataRange))
        branches = starmap_with_kwargs(pool, peak_signal_noise_ratio , args_iter, kwargs_iter)
    finally:
        pool.close()
        pool.join()

    return sum(branches)/dataset_original.shape[0]
# start = time.time()
# a = psnr_wholedataset_pool(t1ce_data, utils.rician_wholedataset(t1ce_data, 5,3,240*240))
# # Grab Currrent Time After Running the Code
# end = time.time()

# #Subtract Start Time from The End Time
# total_time = end - start
# print(total_time)

# start = time.time()
# a = psnr_wholedataset(t1ce_data, utils.rician_wholedataset(t1ce_data, 5,3,240*240))
# # Grab Currrent Time After Running the Code
# end = time.time()

# #Subtract Start Time from The End Time
# total_time = end - start
# print(total_time)

# dataset_original = t1ce_data
# dataset_denoised = utils.rician_wholedataset(t1ce_data, 5,3,240*240)
# data = zip(list(unstack(dataset_original,0)), list(unstack(dataset_denoised,0)))
# start = time.time()
# a = psnr_wholedataset_pool1(data)
# # Grab Currrent Time After Running the Code
# end = time.time()

# #Subtract Start Time from The End Time
# total_time = end - start
# print(total_time)

# 1.3364076614379883
# 0.9268815517425537
# 0.5205528736114502


def mse_wholedataset(dataset_original, dataset_denoised):

    avgmse = 0
    for i in tqdm(range(dataset_original.shape[0])):
        mse = mean_squared_error(dataset_original[i], dataset_denoised[i])
        avgmse += mse
        avgmse = avgmse/2

    return avgmse


def ssim_wholedataset(dataset_original, dataset_denoised):

    avgssim = 0
    for i in tqdm(range(dataset_original.shape[0])):
        SSIM = ssim(dataset_original[i], dataset_denoised[i], multichannel=True,
                    gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        avgssim += SSIM
        avgssim = avgssim/2

    return avgssim