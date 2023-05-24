# 测试
# 开发时间：2022/9/20 10:38
import lpips
from utils import utils_image as util

class util_of_lpips():
    def __init__(self, net, use_gpu=True):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU，默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        # util.modcrop(img_H, 4)
        # print(type(lpips.load_image(img1_path)))
        # img0 = lpips.im2tensor(util.modcrop(lpips.load_image(img1_path), 4))  # RGB image from [-1,1]
        # img1 = lpips.im2tensor(util.modcrop(lpips.load_image(img2_path), 4))
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))


        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        dist01 = self.loss_fn.forward(img0, img1)
        return dist01
if __name__ == '__main__':
    # ours 0.3897(IR700)  0.3901(CVC09-1K)
    # imgLR = '/home/yk/Compare_models/YK-master/results/HBRT-128-CA-conv1-res-edge-IR700-resultC/result18_HBRT.png'
    imgLR = '/home/yk/Compare_models/YK-master/results/HBRT-128-CA-res-edge0.2-CVC09-1K-resultC/result18_HBRT.png'

    # DPSR 0.4012
    # imgLR = '/home/yk/Compare_models/YK-master/results/DPSR-IR700-resultC/result18.png'

    # swinIR 0.3881
    # imgLR = '/home/yk/Compare_models/YK-master/results/SwinIR-CVC09-1K-resultC/result18.png'

    imgHR = '/home/yk/Compare_models/pytorch-edsr-master/datasets/results-C/result18.png'
    metrix = util_of_lpips('alex') # 'vgg' 'alex'
    out = metrix.calc_lpips(imgLR,imgHR)
    print(out)