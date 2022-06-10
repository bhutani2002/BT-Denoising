from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle, denoise_bilateral,denoise_wavelet, unsupervised_wiener
from scipy.signal import convolve2d as conv2

from skimage.filters import median
from medpy.filter.smoothing import anisotropic_diffusion
import copy
from tqdm import tqdm

import numpy as np
from scipy import ndimage as nd


def GaussianFilter_wholedataset(noised_dataset, sigma):
    gaussian_dataset = copy.deepcopy(noised_dataset)
    for i in tqdm(range(gaussian_dataset.shape[0])):
        gaussian_dataset[i] = nd.gaussian_filter(
            tuple(noised_dataset[i, :, :, :]), sigma=sigma)
    return gaussian_dataset


# gauss = GaussianFilter_wholedataset(rayleigh_added, 2)


def BilateralFilter_wholedataset(noised_dataset, sigma_spatial, channel_axis):
    bilateral_dataset = copy.deepcopy(noised_dataset)
    for i in tqdm(range(bilateral_dataset.shape[0])):
        bilateral_dataset[i] = denoise_bilateral(noised_dataset[i, :, :, :], sigma_spatial=sigma_spatial, channel_axis=channel_axis).reshape(240, 240, 1)
    return bilateral_dataset


# bi = BilateralFilter_wholedataset(rayleigh_added, 15, -1)


def AnisotropicFilter_wholedataset(noised_dataset, niter, kappa, gamma, option):
    anisotropic_dataset = copy.deepcopy(noised_dataset)
    for i in tqdm(range(anisotropic_dataset.shape[0])):
        anisotropic_dataset[i] = anisotropic_diffusion(noised_dataset[i, :, :, :], niter=niter, kappa=kappa, gamma=gamma, option=option).reshape(240, 240, 1)
    return anisotropic_dataset


# ani = AnisotropicFilter_wholedataset(rayleigh_added, 50, 20, 0.2, 1)


def WienerFilter_wholedataset(noised_dataset):
    wiener_dataset = copy.deepcopy(noised_dataset)
    for i in tqdm(range(wiener_dataset.shape[0])):
        rng = np.random.default_rng()
        psf = np.ones((5, 5)) / 2
        t1ce = noised_dataset[i, :, :, :].squeeze()
        t1ce=tuple(t1ce)
        t1ce = conv2(t1ce, psf, 'same')
        t1ce += 0.1 * t1ce.std() * rng.standard_normal(t1ce.shape)
        wiener_dataset[i] = unsupervised_wiener(t1ce, psf)[
            0].reshape(240, 240, 1)
    return wiener_dataset


# wiener = WienerFilter_wholedataset(rayleigh_added)

def nlm_wholedataset(noised_dataset):
    nlm_dataset = copy.deepcopy(noised_dataset)
    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(noised_dataset[0, :, :, :], channel_axis=-1))
    print("Sigma Val:", sigma_est)
    for i in tqdm(range(noised_dataset.shape[0])):
        patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)
        nlm_dataset[i] = denoise_nl_means(noised_dataset[i,:,:,:], h=1.15 * sigma_est, fast_mode=True, **patch_kw).reshape(240, 240, 1)
    return nlm_dataset


def median_wholedataset(noised_dataset):
    median_wholedata = copy.deepcopy(noised_dataset)
    for i in tqdm(range(noised_dataset.shape[0])):
        median_wholedata[i] = median(noised_dataset[i,:,:,:][:,:,0], np.ones((3, 3))).reshape(240,240,1)
    return median_wholedata

def totalvar_wholedataset(noised_dataset):
    totalvar_wholedata = copy.deepcopy(noised_dataset)
    # estimate the noise standard deviation from the noisy image
    sigma_est = np.mean(estimate_sigma(noised_dataset[0, :, :, :], channel_axis=-1))
    for i in tqdm(range(noised_dataset.shape[0])):
        totalvar_wholedata[i] = denoise_tv_chambolle(noised_dataset[i,:,:,:][:,:,0], weight=0.1, channel_axis=-1).reshape(240,240,1)
    return totalvar_wholedata


def bilateral2_wholedataset(noised_dataset):
    bilateral2_wholedata = copy.deepcopy(noised_dataset)
    for i in tqdm(range(noised_dataset.shape[0])):
        bilateral2_wholedata[i] = denoise_bilateral(noised_dataset[i,:,:,:], sigma_color=0.05, sigma_spatial=15,channel_axis=-1).reshape(240,240,1)
    return bilateral2_wholedata


def wavelet_wholedataset(noised_dataset):
    wavelet_wholedata = copy.deepcopy(noised_dataset)
    for i in tqdm(range(noised_dataset.shape[0])):
        wavelet_wholedata[i] = denoise_wavelet(noised_dataset[i,:,:,:][:,:,0], channel_axis=-1, rescale_sigma=True).reshape(240,240,1)
    return wavelet_wholedata