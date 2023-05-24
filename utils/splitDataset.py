# 测试
# 开发时间：2023/2/12 10:34
import os
import shutil
import random

import numpy as np
nums = np.ones(700)
nums[:140] = 0
np.random.shuffle(nums)

index = 0

# for path, fileDirs, files in os.walk('/home/yk/Compare_models/YK-master/datasets/CVC-09-1K/val'):
#     for fileName in files:
#         index = index+1
# print(index)
for path, fileDirs, files in os.walk('/home/yk/Compare_models/YK-master/datasets/IR700/train'):
    for fileName in files:
        if nums[index] == 1 :
            shutil.copy(os.path.join(path,fileName),'/home/yk/Compare_models/YK-master/datasets/IR700/train/train_H')
        elif nums[index] == 0:
            shutil.copy(os.path.join(path,fileName),'/home/yk/Compare_models/YK-master/datasets/IR700/train/valid_H')
        index = index+1

# for path, fileDirs, files in os.walk('/home/yk/Compare_models/YK-master/datasets/BSD100/image_SRF_4'):
#     for fileName in files:
#         print(fileName)
#         if fileName.find('HR') !=-1:
#             shutil.copy(os.path.join(path, fileName), '/home/yk/Compare_models/YK-master/datasets/BSD100/image_SRF_4/HR')
#         if fileName.find('LR') !=-1:
#             shutil.copy(os.path.join(path, fileName),'/home/yk/Compare_models/YK-master/datasets/BSD100/image_SRF_4/LR')
