import random
import numpy as np
import torch
import torch.utils.data as data
import utilss.utils_image as util
from utils import utils_blindsr as blindsr

class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = 96
        # self.opt['phase'] == 'train' and
        if self.opt['phase'] == 'train' and len(self.opt['H_size']) == 1:
            self.patch_size = self.opt['H_size'][0] if self.opt['H_size'][0] else 96
            self.L_size = self.patch_size // self.sf
        elif self.opt['phase'] == 'train' and len(self.opt['H_size']) == 2:
            # H W
            self.H_patch_size = self.opt['H_size'][0]
            self.W_patch_size = self.opt['H_size'][1]
            self.H_L_size = self.H_patch_size // self.sf
            self.W_L_size = self.W_patch_size // self.sf
        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = ''
        if opt['dataroot_L']:
            self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------

            # 这里可以尝试加BSRGAN的退化处理,从而生成多样化的img_L
            # --------------------------------
            # img_L, img_H = blindsr.degradation_bsrgan(img_H, self.sf, 120)

            # H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            if len(self.opt['H_size']) == 1:
                rnd_h = random.randint(0, max(0, H - self.L_size))
                rnd_w = random.randint(0, max(0, W - self.L_size))
                img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
                # --------------------------------
                # crop corresponding H patch
                # --------------------------------
                rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
                img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            elif len(self.opt['H_size']) == 2:
                rnd_h = random.randint(0, max(0, H - self.H_L_size))
                rnd_w = random.randint(0, max(0, W - self.W_L_size))
                img_L = img_L[rnd_h:rnd_h + self.H_L_size, rnd_w:rnd_w + self.W_L_size, :]
                # --------------------------------
                # crop corresponding H patch
                # --------------------------------
                rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
                img_H = img_H[rnd_h_H:rnd_h_H + self.H_patch_size, rnd_w_H:rnd_w_H + self.W_patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)
        # else:
        #     H, W, C = img_L.shape
        #
        #     if len(self.opt['H_size']) == 1:
        #         rnd_h = random.randint(0, max(0, H - self.L_size))
        #         rnd_w = random.randint(0, max(0, W - self.L_size))
        #         img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
        #         # --------------------------------
        #         # crop corresponding H patch
        #         # --------------------------------
        #         rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        #         img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]
        #
        #     elif len(self.opt['H_size']) == 2:
        #         rnd_h = random.randint(0, max(0, H - self.H_L_size))
        #         rnd_w = random.randint(0, max(0, W - self.W_L_size))
        #         img_L = img_L[rnd_h:rnd_h + self.H_L_size, rnd_w:rnd_w + self.W_L_size, :]
        #         # --------------------------------
        #         # crop corresponding H patch
        #         # --------------------------------
        #         rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
        #         img_H = img_H[rnd_h_H:rnd_h_H + self.H_patch_size, rnd_w_H:rnd_w_H + self.W_patch_size, :]
        #
        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)

