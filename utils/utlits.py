import math
import shutil

import numpy as np
import torch
import logging
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_region(feature, fixed_size=(14, 14)):
    avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)).cuda()
    B, C, H, W = feature.shape
    region_map = []

    for b in range(B):
        feature_map = feature[b]

        # data = (feature.cpu().detach().numpy())
        data = feature_map.clone().detach()
        heatmap = data.sum(0) / data.shape[0]
        # print(heatmap.shape)
        # heatmap = cv2.resize(heatmap.cpu().numpy(), (H, W))
        # i, j = np.unravel_index(heatmap.argmax(), heatmap.shape)
        heatmap = torch.reshape(heatmap, shape=(H, W))
        temp = torch.zeros_like(heatmap)
        kernel = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        kernel.requires_grad = False
        heatmap = F.conv2d(heatmap.unsqueeze(0).unsqueeze(0), kernel)

        heatmap = heatmap.squeeze(0).squeeze(0)

        # temp[1:27, 1:27] = heatmap
        temp[1:13, 1:13] = heatmap
        i, j = np.unravel_index(temp.cpu().numpy().argmax(), temp.shape)
        # todo
        output = []
        max_H, max_W = feature_map.shape[1] , feature_map.shape[2]   # 最大宽，长
        feature_map_s = H * W  # 原图面积
        w_h_ratio = np.array([1, 0.5, 2])  # 不同长宽比
        area_ratio = np.array([feature_map_s * (1 / 2), feature_map_s * (1 / 3), feature_map_s * (2 / 3)])  # 不同面积比例
        # area_ratio = np.array([feature_map_s * (1 / 3), feature_map_s * (2 / 5)])  # 不同面积比例
        for s in area_ratio:
            for ratio in w_h_ratio:
                h = int(np.sqrt(s / ratio))
                w = int(s // h)
                if (i - h // 2) < 0 and (i + h // 2 + 1) <= max_H and (j - w // 2) < 0 and (j + w // 2 + 1) <= max_W:
                    region = (feature_map[:, 0:min(i + h // 2 + 1,max_H), 0:min(j + w // 2 + 1,max_W)]).unsqueeze(0)
                    # print("case 1 =====")
                    # print(f"h={h>14}")
                    # print(f"w={w>14}")
                elif (i - h // 2) < 0 and (i + h // 2 + 1) <= max_H and (j - w // 2) >= 0 and (j + w // 2 + 1) <= max_W:
                    region = (feature_map[:, 0:min(i + h // 2 + 1,max_H), (j - w // 2):(j + w // 2 + 1)]).unsqueeze(0)
                    # print("case 2 =====")
                    # print(f"h={h>14}")
                    # print(f"(j - w // 2)={(j - w // 2)<0}")
                    # print(f"(j + w // 2)={(j + w // 2)>14}")
                elif (i - h // 2) < 0 and (i + h // 2 + 1) <= max_H and (j - w // 2) >= 0 and (j + w // 2 + 1) > max_W:
                    region = (feature_map[:, 0:min(i + h // 2 + 1,max_H), max((j - w // 2),0):max_W]).unsqueeze(0)
                    # print("case 3 =====")
                    # print(f"h={h>14}")
                elif (i - h // 2) >= 0 and (i + h // 2 + 1) <= max_H and (j - w // 2) >= 0 and (j + w // 2 + 1) > max_W:
                    region = (feature_map[:, (i - h // 2):(i + h // 2 + 1), max((j - w // 2),0):max_W]).unsqueeze(0)
                    # print("case 4 =====")
                    # print(f"(i - h // 2)={(i - h // 2)<0}")
                    # print(f"(i + h // 2)={(i + h // 2)>14}")
                elif (i - h // 2) >= 0 and (i + h // 2 + 1) > max_H and (j - w // 2) >= 0 and (j + w // 2 + 1) > max_W:
                    region = (feature_map[:, max((i - h // 2), 0):max_H, max((j - w // 2), 0):max_W]).unsqueeze(0)
                    # print("case 5 =====")
                elif (i - h // 2) >= 0 and (i + h // 2 + 1) > max_H and (j - w // 2) >= 0 and (j + w // 2 + 1) <= max_W:
                    region = (feature_map[:, max((i - h // 2), 0):max_H, (j - w // 2):(j + w // 2 + 1)]).unsqueeze(0)
                    # print("case 6 =====")
                    # print(f"(j - w // 2)={(j - w // 2)<0}")
                    # print(f"(j + w // 2)={(j + w // 2)>14}")
                elif (i - h // 2) >= 0 and (i + h // 2 + 1) > max_H and (j - w // 2) < 0 and (j + w // 2 + 1) <= max_W:
                    region = (feature_map[:, max((i - h // 2), 0):max_H, 0:min(j + w // 2 + 1,max_W)]).unsqueeze(0)
                    # print("case 7 =====")
                    # print(f"w={w>14}")
                elif (i - h // 2) >= 0 and (i + h // 2 + 1) <= max_H and (j - w // 2) < 0 and (j + w // 2 + 1) <= max_W:
                    region = (feature_map[:, (i - h // 2):(i + h // 2 + 1), 0:min(j + w // 2 + 1,max_W)]).unsqueeze(0)
                    # print("case 8 =====")
                    # print(f"(i - h // 2)={(i - h // 2)<0}")
                    # print(f"(i + h // 2)={(i + h // 2)>14}")
                    # print(f"w={w>14}")
                else:
                    region = (feature_map[:, max((i - h // 2),0):min((i + h // 2 + 1),max_H), max((j - w // 2),0):min((j + w // 2 + 1)
                                                                                                                      ,max_W)]).unsqueeze(0)
                #     print(f"case 0 =====")
                # print(f"region shape = {region.shape}")
                region = region.clone()
                # print(f"region.shape={region.shape}")
                region = torch.squeeze(F.interpolate(region, size=[fixed_size[0], fixed_size[1]], mode="bilinear", align_corners=True), dim=0)
                # print(f"after region shape = {region.shape}")
                region = (avg(region)).view(C)
                output.append(region)

        region_map.append(torch.stack(output))
    region_map = torch.stack(region_map)
    # print(region_map.shape)
    return region_map
# def get_region(feature, fixed_size=(14, 14)):
#     avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)).cuda()
#     B, C, H, W = feature.shape
#     region_map = []
#
#     for b in range(B):
#         feature_map = feature[b]
#
#         # data = (feature.cpu().detach().numpy())
#         data = feature_map.clone()
#         heatmap = data.sum(0) / data.shape[0]
#         heatmap = torch.maximum(heatmap, torch.tensor(0).cuda())
#         heatmap /= torch.max(heatmap)
#         # heatmap = cv2.resize(heatmap.cpu().numpy(), (H, W))
#         # i, j = np.unravel_index(heatmap.argmax(), heatmap.shape)
#         heatmap = torch.reshape(heatmap, shape=(H, W))
#
#         i, j = np.unravel_index(heatmap.detach().cpu().numpy().argmax(), heatmap.shape)
#         output = []
#         max_H, max_W = feature_map.shape[1] , feature_map.shape[2]   # 最大宽，长
#         feature_map_s = H * W  # 原图面积
#         w_h_ratio = np.array([1, 0.5, 2])  # 不同长宽比
#         area_ratio = np.array([feature_map_s * (1 / 2), feature_map_s * (1 / 3), feature_map_s * (2 / 3)])  # 不同面积比例
#         for s in area_ratio:
#             for ratio in w_h_ratio:
#                 h = np.int(np.sqrt(s / ratio))
#                 w = np.int(s // h)
#                 if (i - h // 2) < 0 and (i + h // 2) <= max_H and (j - w // 2) < 0 and (j + w // 2) <= max_W:
#                     region = (feature_map[:, 0:min(h,max_H), 0:min(w,max_W)]).unsqueeze(0)
#                     # print("case 1 =====")
#                     # print(f"h={h>14}")
#                     # print(f"w={w>14}")
#                 elif (i - h // 2) < 0 and (i + h // 2) <= max_H and (j - w // 2) >= 0 and (j + w // 2) <= max_W:
#                     region = (feature_map[:, 0:min(h,max_H), (j - w // 2):(j + w // 2)]).unsqueeze(0)
#                     # print("case 2 =====")
#                     # print(f"h={h>14}")
#                     # print(f"(j - w // 2)={(j - w // 2)<0}")
#                     # print(f"(j + w // 2)={(j + w // 2)>14}")
#                 elif (i - h // 2) < 0 and (i + h // 2) <= max_H and (j - w // 2) >= 0 and (j + w // 2) > max_W:
#                     region = (feature_map[:, 0:min(h,max_H), max((max_W - w),0):max_W]).unsqueeze(0)
#                     # print("case 3 =====")
#                     # print(f"h={h>14}")
#                 elif (i - h // 2) >= 0 and (i + h // 2) <= max_H and (j - w // 2) >= 0 and (j + w // 2) > max_W:
#                     region = (feature_map[:, (i - h // 2):(i + h // 2), max((max_W - w),0):max_W]).unsqueeze(0)
#                     # print("case 4 =====")
#                     # print(f"(i - h // 2)={(i - h // 2)<0}")
#                     # print(f"(i + h // 2)={(i + h // 2)>14}")
#                 elif (i - h // 2) >= 0 and (i + h // 2) > max_H and (j - w // 2) >= 0 and (j + w // 2) > max_W:
#                     region = (feature_map[:, max((max_H - h), 0):max_H, max((max_W - w), 0):max_W]).unsqueeze(0)
#                     # print("case 5 =====")
#                 elif (i - h // 2) >= 0 and (i + h // 2) > max_H and (j - w // 2) >= 0 and (j + w // 2) <= max_W:
#                     region = (feature_map[:, max((max_H - h), 0):max_H, (j - w // 2):(j + w // 2)]).unsqueeze(0)
#                     # print("case 6 =====")
#                     # print(f"(j - w // 2)={(j - w // 2)<0}")
#                     # print(f"(j + w // 2)={(j + w // 2)>14}")
#                 elif (i - h // 2) >= 0 and (i + h // 2) > max_H and (j - w // 2) < 0 and (j + w // 2) <= max_W:
#                     region = (feature_map[:, max((max_H - h), 0):max_H, 0:min(w,max_W)]).unsqueeze(0)
#                     # print("case 7 =====")
#                     # print(f"w={w>14}")
#                 elif (i - h // 2) >= 0 and (i + h // 2) <= max_H and (j - w // 2) < 0 and (j + w // 2) <= max_W:
#                     region = (feature_map[:, (i - h // 2):(i + h // 2), 0:min(w,max_W)]).unsqueeze(0)
#                     # print("case 8 =====")
#                     # print(f"(i - h // 2)={(i - h // 2)<0}")
#                     # print(f"(i + h // 2)={(i + h // 2)>14}")
#                     # print(f"w={w>14}")
#                 else:
#                     region = (feature_map[:, max((i - h // 2),0):min((i + h // 2),max_H), max((j - w // 2),0):min((j + w // 2),max_W)]).unsqueeze(0)
#                     # print(f"case 0 =====")
#                 region = region.clone()
#                 # print(f"region.shape={region.shape}")
#                 region = torch.squeeze(F.interpolate(region, size=[fixed_size[0], fixed_size[1]], mode="bilinear"), dim=0)
#                 region = (avg(region)).view(C)
#                 output.append(region)
#
#         region_map.append(torch.stack(output))
#     region_map = torch.stack(region_map)
#     # print(region_map.shape)
#     return region_map


def save_checkpoint(state, is_best, args, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.pth'.format(args.data_type))


def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)

    return mx


def show_args(args):
    logging.info("==========================================")
    logging.info("==========       CONFIG      =============")
    logging.info("==========================================")

    for arg, content in args.__dict__.items():
        logging.info("{}: {}".format(arg, content))

    logging.info("==========================================")
    logging.info("===========        END        ============")
    logging.info("==========================================")

    logging.info("\n")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def randomint_plus(low, high=None, cutoff=None, size=None):
    """用来生成不包含一些中间数值的随机整数或序列

    Parameters
    ----------
    low : int
        生成随机整数的下界
    high: int
        生成随机整数的上界
    cutoff: int/list
        不需要包含的一个或多个值
    size: tuple
        需要产生的随机数规模
    Notes
    -----
    1. 在调用过程中，如果high, cutoff, size都缺省，就默认返回一个[0, low)的值
    2. 如果cutoff, size缺省，返回[low, high)的一个随机整数值
    3. 如果size缺省， 则返回一个[low, cutoff)U(cutoff, high)的随机整数值
    4. 返回一个给定size的矩阵，矩阵元素为[low, cutoff)U(cutoff, high)的随机整数值

    See Also
    --------
    np.random.randint()

    """
    if high is None:
        assert low is not None, "必须给定一个值作为上界"
        high = low
        low = 0
    number = 1  # 将size拉长成为一个向量
    if size is not None:
        for i in range(len(size)):
            number = number * size[i]

    if cutoff is None:  # 如果不需要剔除值，就通过numpy提供的函数生成随机整数
        random_result = np.random.randint(low, high, size=size)
    else:
        if number == 1:  # 返回一个随机整数
            random_result = randint_digit(low, high, cutoff)
        else:  # 返回一个形状为size的随机整数数组
            random_result = np.zeros(number, )
            for i in range(number):
                random_result[i] = randint_digit(low, high, cutoff)
            random_result = random_result.reshape(size)

    return random_result.astype(int)


def randint_digit(low, high, cutoff):
    """用来生成一个在区间[low, high)排除cutoff后的随机整数

    Parameters
    ----------
    low: int
        下限，能够取到
    high: int
        上限，不能够取到
    cutoff: int/list
        一个需要剔除的数或者数组，要求在(low, high)区间之间
    """
    digit_list = list(range(low, high))
    if type(cutoff) is int:  # 只需要剔除一个值
        if cutoff in digit_list:  # 如果需要剔除的值不存在，则不执行剔除操作
            digit_list.remove(cutoff)
    else:
        for i in cutoff:  # 需要剔除多个值的情况
            if i not in digit_list:  # 如果需要剔除的值不存在，则不执行剔除操作
                continue
            digit_list.remove(i)

    np.random.shuffle(digit_list)

    return digit_list.pop()  # 生成的序列打乱并且返回当前的随机值