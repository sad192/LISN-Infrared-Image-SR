# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2
import torch

from utils import utils_image as util
from torchvision.transforms.functional_tensor import rgb_to_grayscale

import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth




"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# From 2019/03--2021/08
# --------------------------------------------
"""

def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor

    Return:
        cropped image
    '''
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[:w - w % sf, :h - h % sf, ...]


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""
def analytic_kernel(k):
    """Calculate the X4 kernel from the X2 kernel (for proof see appendix in paper)"""
    k_size = k.shape[0]
    # Calculate the big kernels size
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    # Loop over the small kernel to fill the big one
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    # Crop the edges of the big kernel to ignore very small values and increase run time of SR
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    # Normalize to 1
    return cropped_big_k / cropped_big_k.sum()


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def blur(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, Nx1xhxw
    '''
    n, c = x.shape[:2]
    p1, p2 = (k.shape[-2]-1)//2, (k.shape[-1]-1)//2
    x = torch.nn.functional.pad(x, pad=(p1, p2, p1, p2), mode='replicate')
    k = k.repeat(1,c,1,1)
    k = k.view(-1, 1, k.shape[2], k.shape[3])
    x = x.view(1, -1, x.shape[2], x.shape[3])
    x = torch.nn.functional.conv2d(x, k, bias=None, stride=1, padding=0, groups=n*c)
    x = x.view(n, c, x.shape[2], x.shape[3])

    return x



def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h


def fspecial_laplacian(alpha):
    alpha = max([0, min([alpha,1])])
    h1 = alpha/(alpha+1)
    h2 = (1-alpha)/(alpha+1)
    h = [[h1, h2, h1], [h2, -4/(alpha+1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)
    if filter_type == 'laplacian':
        return fspecial_laplacian(*args, **kwargs)

"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    '''
    x = util.imresize_np(x, scale=1/sf)
    return x


def srmd_degradation(x, k, sf=3):
    ''' blur + bicubic downsampling

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):

    ''' bicubic downsampling + blur

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    '''
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def add_sharpening(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening. borrowed from real-ESRGAN
    Input image: I; Blurry image: B.
    1. K = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * K + (1 - Mask) * I
    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = img + weight * residual
    K = np.clip(K, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


def add_blur(img, sf=4):
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2*sf
    if random.random() < 0.5:
        l1 = wd2*random.random()
        l2 = wd2*random.random()
        ## ksize=2*random.randint(2,11)+3
        k = anisotropic_Gaussian(ksize=random.randint(3,9), theta=random.random()*np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', random.randint(3,9), wd*random.random())
    img = ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5/sf, 1)
    else:
        sf1 = 1.0
    img = cv2.resize(img, (int(sf1*img.shape[1]), int(sf1*img.shape[0])), interpolation=random.choice([1, 2, 3]))
    img = np.clip(img, 0.0, 1.0)

    return img


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:   # add color Gaussian noise
        img += np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4: # add grayscale Gaussian noise
        img += np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:            # add  noise
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def add_speckle_noise(img, noise_level1=2, noise_level2=25):
    noise_level = random.randint(noise_level1, noise_level2)
    img = np.clip(img, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        img += img*np.random.normal(0, noise_level/255.0, img.shape).astype(np.float32)
    elif rnum < 0.4:
        img += img*np.random.normal(0, noise_level/255.0, (*img.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2/255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3,3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        img += img*np.random.multivariate_normal([0,0,0], np.abs(L**2*conv), img.shape[:2]).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)
    return img


def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    """Generate a batch of poisson noise (PyTorch version)

    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0
    if cal_gray_noise:
        img_gray = rgb_to_grayscale(img, num_output_channels=1)
        # round and clip image for counting vals correctly
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.
        # use for-loop to get the unique values for each sample
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = out - img_gray
        noise_gray = noise_gray.expand(b, 3, h, w)

    # always calculate color noise
    # round and clip image for counting vals correctly
    img = torch.clamp((img * 255.0).round(), 0, 255) / 255.
    # use for-loop to get the unique values for each sample
    vals_list = [len(torch.unique(img[i, :, :, :])) for i in range(b)]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = img.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img * vals) / vals
    noise = out - img
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def add_poisson_noise_pt(img, scale=1.0, clip=True, rounds=False, gray_noise=0):
    """Add poisson noise to a batch of images (PyTorch version).

    Args:
        img (Tensor): Input image, shape (b, c, h, w), range [0, 1], float32.
        scale (float | Tensor): Noise scale. Number or Tensor with shape (b).
            Default: 1.0.
        gray_noise (float | Tensor): 0-1 number or Tensor with shape (b).
            0 for False, 1 for True. Default: 0.

    Returns:
        (Tensor): Returned noisy image, shape (b, c, h, w), range[0, 1],
            float32.
    """
    noise = generate_poisson_noise_pt(img, scale, gray_noise)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def add_Poisson_noise(img):
    img = np.clip((img * 255.0).round(), 0, 255) / 255.
    vals = 10**(2*random.random()+2.0)  # [2, 4]
    if random.random() < 0.5:
        img = np.random.poisson(img * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        img += noise_gray[:, :, np.newaxis]
    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img):
    quality_factor = random.randint(30, 80)
    img = cv2.cvtColor(util.single2uint(img), cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    img = cv2.imdecode(encimg, 1)
    img = cv2.cvtColor(util.uint2single(img), cv2.COLOR_BGR2RGB)
    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    h, w = lq.shape[:2]
    # rnd_h = random.randint(0, h-lq_patchsize)
    # rnd_w = random.randint(0, w-lq_patchsize)
    nums_h = list(range(0, int(h-lq_patchsize+1)))
    rnd_h_arr = random.sample(nums_h, 1)
    rnd_h = rnd_h_arr[0]
    nums_w = list(range(0, int(w - lq_patchsize + 1)))
    rnd_w_arr = random.sample(nums_w, 1)
    rnd_w = rnd_w_arr[0]

    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize*sf, rnd_w_H:rnd_w_H + lq_patchsize*sf, :]
    return lq, hq

def random_crop_test(lq, hq, sf=4, h_lq=64,w_lq=64):
    h, w = lq.shape[:2]
    # rnd_h = random.randint(0, h-lq_patchsize)
    # rnd_w = random.randint(0, w-lq_patchsize)
    nums_h = list(range(0, int(h-h_lq+1)))
    rnd_h_arr = random.sample(nums_h, 1)
    rnd_h = rnd_h_arr[0]
    nums_w = list(range(0, int(w - w_lq + 1)))
    rnd_w_arr = random.sample(nums_w, 1)
    rnd_w = rnd_w_arr[0]

    lq = lq[rnd_h:rnd_h + h_lq, rnd_w:rnd_w + w_lq, :]

    # rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    # hq = hq[rnd_h_H:rnd_h_H + h_lq*sf, rnd_w_H:rnd_w_H + lq_patchsize*sf, :]
    return lq, hq


def degradation_bsrgan(img, sf=4, lq_patchsize=72, isp_model=None):
    """
    This is the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    isp_model: camera ISP model

    Returns
    -------
    img: low-quality patch, size: lq_patchsize X lq_patchsize X C, range: [0, 1] 像素值都做了归一化
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """
    # 各个因子的概率：各向同性高斯核 放大因子。实现随机下采样
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf # 放大因子
    # print('输入图片：',img.shape)
    h1, w1 = img.shape[:2]  # 获取高分辨率图的高宽
    # 根据 C H W 还是 C W H
    # img = img.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]  # mod crop 裁剪高宽
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]
    h, w = img.shape[:2]  # 获取裁剪后的高宽

    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    hq = img.copy()  # 若高宽和缩放因子成比例则不会裁剪

    if sf == 4 and random.random() < scale2_prob:   # downsample1
        if np.random.rand() < 0.5:
            # 将 img(H) 缩放1/2 插值算法类型随机
            img = cv2.resize(img, (int(1/2*img.shape[1]), int(1/2*img.shape[0])), interpolation=random.choice([1,2,3]))
        else:
            img = util.imresize_np(img, 1/2, True)  # ?
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # keep downsample3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:

        if i == 0:
            img = add_blur(img, sf=sf)

        elif i == 1:
            img = add_blur(img, sf=sf)

        elif i == 2:
            a, b = img.shape[1], img.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1,2*sf)
                img = cv2.resize(img, (int(1/sf1*img.shape[1]), int(1/sf1*img.shape[0])), interpolation=random.choice([1,2,3]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6*sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted/k_shifted.sum()  # blur with shifted kernel
                img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]  # nearest downsampling
            img = np.clip(img, 0.0, 1.0)

        elif i == 3:
            # downsample3
            img = cv2.resize(img, (int(1/sf*a), int(1/sf*b)), interpolation=random.choice([1,2,3]))
            img = np.clip(img, 0.0, 1.0)

        elif i == 4:
            # add Gaussian noise
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)

        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)

        elif i == 6:
            # add processed camera sensor noise
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    # img, hq = random_crop(img, hq, sf_ori, lq_patchsize)

    return img, hq




def degradation_bsrgan_plus(img, sf=4, shuffle_prob=0.5, use_sharp=False, lq_patchsize=64, isp_model=None):
    """
    This is an extended degradation model by combining
    the degradation models of BSRGAN and Real-ESRGAN
    ----------
    img: HXWXC, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    use_shuffle: the degradation shuffle
    use_sharp: sharpening the img

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsizeXC, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf)XC, range: [0, 1]
    """

    h1, w1 = img.shape[:2]
    ## test不裁剪
    # img = img.copy()[:w1 - w1 % sf, :h1 - h1 % sf, ...]  # mod crop
    h, w = img.shape[:2]
    h_lq = h // sf
    w_lq = w // sf
    if h < lq_patchsize*sf or w < lq_patchsize*sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    if use_sharp:
        img = add_sharpening(img)
    hq = img.copy()
    # print('hq',hq.shape)
    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(13), 13)
    else:
        shuffle_order = list(range(13))
        # local shuffle for noise, JPEG is always the last one
        shuffle_order[2:6] = random.sample(shuffle_order[2:6], len(range(2, 6)))
        shuffle_order[9:13] = random.sample(shuffle_order[9:13], len(range(9, 13)))

    poisson_prob, speckle_prob, isp_prob = 0.1, 0.1, 0.1

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_resize(img, sf=sf)
        elif i == 2:
            img = add_Gaussian_noise(img, noise_level1=1, noise_level2=30)
        elif i == 3:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img) ##add_Poisson_noise
        elif i == 4:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 5:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        elif i == 6:
            img = add_JPEG_noise(img)
        elif i == 7:
            img = add_blur(img, sf=sf)
        elif i == 8:
            img = add_resize(img, sf=sf)
        elif i == 9:
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 10:
            if random.random() < poisson_prob:
                img = add_Poisson_noise(img)
        elif i == 11:
            if random.random() < speckle_prob:
                img = add_speckle_noise(img)
        elif i == 12:
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)
        else:
            print('check the shuffle!')

    # resize to desired size
    img = cv2.resize(img, (int(1/sf*hq.shape[1]), int(1/sf*hq.shape[0])), interpolation=random.choice([1, 2, 3]))

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    # img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img, hq



if __name__ == '__main__':
    H_path = '/home/yk/Compare_models/Image-Super-Resolution-via-Iterative-Refinement-master/dataset/640_flir_hr'
    for imgH in util.get_image_paths(H_path):
        img = util.imread_uint(imgH, 3)
        img_name, ext = os.path.splitext(os.path.basename(imgH))

        img1 = util.uint2single(img)
        print('形状：', img1.shape)
        # sf = 2
        # img_L = util.modcrop(img1, sf)
        # img_L = util.imresize_np(img_L, 1 / sf)
        # util.imsave(util.single2uint(img_L),'/home/yk/Compare_models/YK-master/datasets/x2-results-A'++img_name+'-'+'.png')

        sf = 4
        # L_size = 256 // sf
        H, W, C = img1.shape
        # H = H // sf
        # W = W // sf
        # if True:
        #     rnd_h = random.randint(0, max(0, H - L_size))
        #     rnd_w = random.randint(0, max(0, W - L_size))
        #     # img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
        #     # --------------------------------
        #     # crop corresponding H patch
        #     # --------------------------------
        #     rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
        #     img1 = img1[rnd_h_H:rnd_h_H + 256, rnd_w_H:rnd_w_H + 256, :]

        # img1 = cv2.resize(img1, (256, 256), interpolation=cv2.INTER_CUBIC)
        # print('形状1：', img1.shape)

        for i in range(5):
            img_lq, img_hq = degradation_bsrgan_plus(img1, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
            print('退化操作后lq_img: ', img_lq.shape)
            print('img_hq: ', img_hq.shape)
            # sr = cv2.resize(img_lq, (256, 256), interpolation=cv2.INTER_CUBIC)
            sr = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)
            print('img_sr: ', sr.shape)
            # break
            #  最近邻近插值
            # lq_nearest = cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
            # img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1) lr_64 sr_64_256 hr_256
            util.imsave(util.single2uint(img_lq),
                        '/home/yk/Compare_models/Image-Super-Resolution-via-Iterative-Refinement-master/dataset/640_hr_random/lr/' + img_name + '-' + str(
                            i) + '.jpg')
            util.imsave(util.single2uint(sr),
                        '/home/yk/Compare_models/Image-Super-Resolution-via-Iterative-Refinement-master/dataset/640_hr_random/sr/' + img_name + '-' + str(
                            i) + '.jpg')
            util.imsave(util.single2uint(img_hq),
                        '/home/yk/Compare_models/Image-Super-Resolution-via-Iterative-Refinement-master/dataset/640_hr_random/hr/' + img_name + '-' + str(
                            i) + '.jpg')
            # SR_imgs
            # util.imsave(util.single2uint(img_hq), 'hq' + str(i) + '.bmp')``

    # img = util.imread_uint('8.bmp', 3)
    # print('形状：', img.shape)
    # img1 = util.uint2single(img)
    # print('形状1：', img1.shape)
    # sf = 4
    #
    # for i in range(2):
    #     img_lq, img_hq = degradation_bsrgan_plus(img1, sf=sf)
    #     print(i)
    #     print('退化操作后lq_img: ', img_lq.shape)
    #     print('img_hq: ', img_hq.shape)
    #     #  最近邻近插值
    #     # lq_nearest = cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
    #     # img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
    #     util.imsave(util.single2uint(img_lq), 'lq'+str(i)+'.png')
    #     util.imsave(util.single2uint(img_hq), 'hq' + str(i) + '.png')

#    for i in range(10):
#        img_lq, img_hq = degradation_bsrgan_plus(img, sf=sf, shuffle_prob=0.1, use_sharp=True, lq_patchsize=64)
#        print(i)
#        lq_nearest =  cv2.resize(util.single2uint(img_lq), (int(sf*img_lq.shape[1]), int(sf*img_lq.shape[0])), interpolation=0)
#        img_concat = np.concatenate([lq_nearest, util.single2uint(img_hq)], axis=1)
#        util.imsave(img_concat, str(i)+'.png')

#    run utils/utils_blindsr.py
