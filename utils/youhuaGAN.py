# 测试
# 开发时间：2022/6/30 9:35
import torch

# ----------
# 作者把这个数复制成一张feature map的大小，
# 跟原来的feature map拼在一起送给Discriminator。
# ----------

def minibatch_std(x):
    batch_statistics = (
        torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
    )
    # we take the std for each example (across all channels, and pixels) then we repeat it
    # for a single channel and concatenate it with the image. In this way the discriminator
    # will get information about the variation in the batch/image
    return torch.cat([x, batch_statistics], dim=1)


def miniBatchStdDev(x, subGroupSize=4):
    r"""
    Add a minibatch standard deviation channel to the current layer.
    In other words:
        1) Compute the standard deviation of the feature map over the minibatch
        2) Get the mean, over all pixels and all channels of thsi ValueError
        3) expand the layer and cocatenate it with the input
    Args:
        - x (tensor): previous layer
        - subGroupSize (int): size of the mini-batches on which the standard deviation
        should be computed
    """
    size = x.size()
    subGroupSize = min(size[0], subGroupSize)
    if size[0] % subGroupSize != 0:
        subGroupSize = size[0]
    G = int(size[0] / subGroupSize)
    if subGroupSize > 1:
        y = x.view(-1, subGroupSize, size[1], size[2], size[3])
        y = torch.var(y, 1)
        y = torch.sqrt(y + 1e-8)
        y = y.view(G, -1)
        y = torch.mean(y, 1).view(G, 1)
        y = y.expand(G, size[2]*size[3]).view((G, 1, 1, size[2], size[3]))
        y = y.expand(G, subGroupSize, -1, -1, -1)
        y = y.contiguous().view((-1, 1, size[2], size[3]))
    else:
        y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)

    return torch.cat([x, y], dim=1)


if __name__ == '__main__':
    # 输入数据的batchSize要大于1
    x = torch.rand(2,3, 8, 10)
    # out = minibatch_std(x)
    out = miniBatchStdDev(x)
    print(out.shape)
    print(out)

    ### Assuming this gets you real and fake data

    # Real data
    # x.data.resize_as_(images).copy_(images)
    # y_pred = D(x)
    # y.data.resize_(current_batch_size).fill_(1)

    # Fake data
    # z.data.resize_(current_batch_size, param.z_size, 1, 1).normal_(0, 1)
    # fake = G(z)
    # x_fake.data.resize_(fake.data.size()).copy_(fake.data)
    # y_pred_fake = D(x_fake.detach())  # For generator step do not detach
    # y2.data.resize_(current_batch_size).fill_(0)

    ### Relativistic average LSGAN

    # No activation in discriminator

    # Discriminator loss
    # errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - y) ** 2) + torch.mean(
    #     (y_pred_fake - torch.mean(y_pred) + y) ** 2)) / 2
    # errD.backward()

    # Generator loss (You may want to resample again from real and fake data)
    # errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + y) ** 2) + torch.mean(
    #     (y_pred_fake - torch.mean(y_pred) - y) ** 2)) / 2
    # errG.backward()
