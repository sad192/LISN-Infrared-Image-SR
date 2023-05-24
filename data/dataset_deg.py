# 测试
# 开发时间：2022/7/11 17:23
import numpy as np
import torch.utils.data as data
import utils.utils_image as util

class DatasetDeg(data.Dataset):
    '''
        # -----------------------------------------
        # Get H/L for H_image-to-L_image mapping.
        # Both "paths_L" and "paths_H" are needed.
        # -----------------------------------------
        # 需要测试集图片最为target图片
        # -----------------------------------------
    '''
    def __init__(self,opt):
        super(DatasetDeg, self).__init__()
        print('Get H/L for H_image-to-L_image mapping. 测试集图片作为target.')
        # opt 包含训练集 H图片 和 测试集图片 T 的路径 path
        self.opt = opt
        self.n_channels = 3
        # ------------------------------------
        # get the path of H 和 测试集图片的路径 paths_T
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_T = util.get_image_paths(opt['dataroot_T'])
        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_T, 'Error: T path is empty. 需要测试集图片作为target!'
        # if self.paths_T and self.paths_H:
        #     assert len(self.paths_T) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L),
        #                                                                                    len(self.paths_H))

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        # ------------------------------------
        # get T image
        # ------------------------------------
        T_path = self.paths_T[index]
        img_T = util.imread_uint(T_path, self.n_channels)

        # ------------------------------------
        # if train, get H/T patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':
            return {'T': img_T, 'H': img_H, 'T_path': T_path, 'H_path': H_path}
        else:
            return {'T': '', 'H': img_H, 'T_path': '', 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)