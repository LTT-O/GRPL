import torch
import numpy as np
import torch.nn.functional as F

# def get_region(feature, fixed_size=(14, 14)):
# @Time    :2020/12/12 11:40
# @Author  :korolTung
# @FileName: user_function.py
import numpy as np

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

# if __name__ == '__main__':
#     cutoff1 = 3
#     cutoff2 = [3, 5, 10]
#     low = 1
#     high = 9
#     result = randomint_plus(low, high, cutoff2, size=(1, 5))
#     print(type(result))

def get_region(feature, fixed_size=(14, 14)):
    avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)).cuda()
    B, C, H, W = feature.shape
    region_map = []

    for b in range(B):
        feature_map = feature[b]

        # data = (feature.cpu().detach().numpy())
        data = feature_map.clone()
        # print(data.shape)
        heatmap = data.sum(0) / data.shape[0]
        # print(heatmap.shape)
        heatmap = torch.maximum(heatmap, torch.tensor(0).cuda())
        heatmap /= torch.max(heatmap)
        # print(heatmap.shape)
        # heatmap = cv2.resize(heatmap.cpu().numpy(), (H, W))
        # i, j = np.unravel_index(heatmap.argmax(), heatmap.shape)
        heatmap = torch.reshape(heatmap, shape=(H, W))

        kernel = [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        heatmap = F.conv2d(heatmap.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        heatmap = heatmap.squeeze(0).squeeze(0)
        # print(heatmap.shape)
        i, j = np.unravel_index(heatmap.detach().cpu().numpy().argmax(), heatmap.shape)
        print(f"conv feature = {heatmap}")
        print(f"i = {i}")
        print(f"j = {j}")
        # todo
        output = []
        max_H, max_W = feature_map.shape[1] , feature_map.shape[2]   # 最大宽，长
        feature_map_s = H * W  # 原图面积
        w_h_ratio = np.array([1, 0.5, 2])  # 不同长宽比
        area_ratio = np.array([feature_map_s * (1 / 2), feature_map_s * (1 / 3), feature_map_s * (2 / 3)])  # 不同面积比例
        for s in area_ratio:
            for ratio in w_h_ratio:
                h = np.int_(np.sqrt(s / ratio))
                w = np.int_(s // h)
                if (i - h // 2) < 0 and (i + h // 2) <= max_H and (j - w // 2) < 0 and (j + w // 2) <= max_W:
                    region = (feature_map[:, 0:min(i + h // 2,max_H), 0:min(j + w // 2,max_W)]).unsqueeze(0)
                    # print("case 1 =====")
                    # print(f"h={h>14}")
                    # print(f"w={w>14}")
                elif (i - h // 2) < 0 and (i + h // 2) <= max_H and (j - w // 2) >= 0 and (j + w // 2) <= max_W:
                    region = (feature_map[:, 0:min(i + h // 2,max_H), (j - w // 2):(j + w // 2)]).unsqueeze(0)
                    # print("case 2 =====")
                    # print(f"h={h>14}")
                    # print(f"(j - w // 2)={(j - w // 2)<0}")
                    # print(f"(j + w // 2)={(j + w // 2)>14}")
                elif (i - h // 2) < 0 and (i + h // 2) <= max_H and (j - w // 2) >= 0 and (j + w // 2) > max_W:
                    region = (feature_map[:, 0:min(i + h // 2,max_H), max((j - w // 2),0):max_W]).unsqueeze(0)
                    # print("case 3 =====")
                    # print(f"h={h>14}")
                elif (i - h // 2) >= 0 and (i + h // 2) <= max_H and (j - w // 2) >= 0 and (j + w // 2) > max_W:
                    region = (feature_map[:, (i - h // 2):(i + h // 2), max((j - w // 2),0):max_W]).unsqueeze(0)
                    # print("case 4 =====")
                    # print(f"(i - h // 2)={(i - h // 2)<0}")
                    # print(f"(i + h // 2)={(i + h // 2)>14}")
                elif (i - h // 2) >= 0 and (i + h // 2) > max_H and (j - w // 2) >= 0 and (j + w // 2) > max_W:
                    region = (feature_map[:, max((i - h // 2), 0):max_H, max((j - w // 2), 0):max_W]).unsqueeze(0)
                    # print("case 5 =====")
                elif (i - h // 2) >= 0 and (i + h // 2) > max_H and (j - w // 2) >= 0 and (j + w // 2) <= max_W:
                    region = (feature_map[:, max((i - h // 2), 0):max_H, (j - w // 2):(j + w // 2)]).unsqueeze(0)
                    # print("case 6 =====")
                    # print(f"(j - w // 2)={(j - w // 2)<0}")
                    # print(f"(j + w // 2)={(j + w // 2)>14}")
                elif (i - h // 2) >= 0 and (i + h // 2) > max_H and (j - w // 2) < 0 and (j + w // 2) <= max_W:
                    region = (feature_map[:, max((i - h // 2), 0):max_H, 0:min(j + w // 2,max_W)]).unsqueeze(0)
                    # print("case 7 =====")
                    # print(f"w={w>14}")
                elif (i - h // 2) >= 0 and (i + h // 2) <= max_H and (j - w // 2) < 0 and (j + w // 2) <= max_W:
                    region = (feature_map[:, (i - h // 2):(i + h // 2), 0:min(j + w // 2,max_W)]).unsqueeze(0)
                    # print("case 8 =====")
                    # print(f"(i - h // 2)={(i - h // 2)<0}")
                    # print(f"(i + h // 2)={(i + h // 2)>14}")
                    # print(f"w={w>14}")
                else:
                    region = (feature_map[:, max((i - h // 2),0):min((i + h // 2),max_H), max((j - w // 2),0):min((j + w // 2),max_W)]).unsqueeze(0)
                    # print(f"case 0 =====")
                region = region.clone()
                print(f"region = {region}")
                return
                # print(f"region.shape={region.shape}")
                region = torch.squeeze(F.interpolate(region, size=[fixed_size[0], fixed_size[1]], mode="bilinear"), dim=0)
                region = (avg(region)).view(C)
                output.append(region)

        region_map.append(torch.stack(output))
    region_map = torch.stack(region_map)
    # print(region_map.shape)
    return region_map

def demo():
    kernel = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    data = torch.randint(0, 10, size=(10, 10), dtype=torch.float).unsqueeze(0).unsqueeze(0)
    print(data)
    result = F.conv2d(data, kernel)
    result = result.squeeze(0).squeeze(0)
    print(result)


import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_CAM(model, img_path, save_path, transform=None, visual_heatmap=False):
    '''
    绘制 Class Activation Map
    :param model: 加载好权重的Pytorch model
    :param img_path: 测试图片路径
    :param save_path: CAM结果保存路径
    :param transform: 输入图像预处理方法
    :param visual_heatmap: 是否可视化原始heatmap（调用matplotlib）
    :return:
    '''
    # 图像加载&预处理
    img = Image.open(img_path)
    if transform:
        img = transform(img).cuda()
    img = img.unsqueeze(0)

    # 获取模型输出的feature/score
    model.eval()
    model.cuda()
    # features = model.features(img)
    # output = model.classifier(features)
    # batch, None, is_eval = True
    output, features = model([img, img, None], None, True)
    print(features.shape)
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    # 预测得分最高的那一类对应的输出score
    pred = torch.argmax(output).item()
    pred_class = output[:, pred]

    features.register_hook(extract)
    pred_class.backward()  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    features = features[0]
    # 512是最后一层feature的通道数
    print(features.shape)
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap + img * 0.5  # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘


def main():
    feature = torch.randint(0, 10, size=(1, 1, 10, 10)).cuda()
    print(f"feature = {feature}")
    # feature[0,0,2,2] = 1
    # feature[0, 0, 3, 2] = 10
    result = get_region(feature, fixed_size=(5, 5))
    # print(result.shape)
    target = np.array([0, 1, 2, 3, 4])
#
#
#
if __name__ == '__main__':
    main()
    # demo()
