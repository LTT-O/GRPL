import cv2
import numpy as np
import torchvision
import torch
from PIL import Image
from model.model_label import MoProGCN
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
    # print(features.shape)
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
    # print(features.shape)
    for i in range(2048):
        features[i, ...] *= pooled_grads[i, ...]

    # 以下部分同Keras版实现
    heatmap = features.cpu().detach().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    i, j = np.unravel_index(heatmap.argmax(), heatmap.shape)
    # print(i, j)
    # 可视化原始热力图
    if visual_heatmap:
        plt.matshow(heatmap)
        plt.show()

    img = cv2.imread(img_path)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    superimposed_img = heatmap * 0.6 + img * 0.4   # 这里的0.4是热力图强度因子
    cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘
CUDA_VISIBLE_DEVICES=1

def main():
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(512, 512)),
        torchvision.transforms.CenterCrop(size=(448, 448)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    img_path = "img/bird/2.jpg"
    save_path = "heatmap/bird/2.jpg"
    img = cv2.imread(img_path)
    # print(img.shape)
    input_img = test_transform(Image.open(img_path)).unsqueeze(0).cuda()
    model = MoProGCN(inputDim=2048, nodeNum=10, num_classes=200)
    model.load_state_dict({k.replace('module.',''):v for k, v in torch.load("exp/model/best_bird-gcn-label-0310_model.pth").items()})
    draw_CAM(model, img_path, save_path, test_transform, True)
    # model.cuda()
    # model.eval()
    # batch = [input_img, input_img, None]
    # output, feature = model(batch, None, is_eval=True)

    #
    # feature = feature.squeeze()
    # show_heatmap(feature, ("heatmap/aircraft/region/0303-node7-epoch34-2.jpg"), img)


if __name__ == '__main__':
    main()

