import torch.nn as nn
import torch.nn.functional as F
import torch
from GCN.layer import GraphConvolution


class GCN(nn.Module):
    def __init__(self, inputDim, hidDim, outputDim, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数

        self.gc1 = GraphConvolution(inputDim, hidDim)  # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(hidDim, outputDim)  # gc2输入尺寸nhid，输出尺寸ncalss
        # self.gc1 = GraphConvolution(inputDim, hidDim)  # gc1输入尺寸nfeat，输出尺寸nhid
        # self.gc2 = GraphConvolution(hidDim, hidDim)  # gc2输入尺寸nhid，输出尺寸ncalss
        # self.gc3 = GraphConvolution(hidDim, outputDim)  # gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout
# 你看吧
    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        batchSize, nodeNum = x.shape[0], adj.shape[0]
        adj = adj.repeat(batchSize, 1).view(batchSize, nodeNum, -1)
        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = self.gc2(x, adj)
        # batchSize, nodeNum = x.shape[0], adj.shape[0]
        # adj = adj.repeat(batchSize, 1).view(batchSize, nodeNum, -1)
        # x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        # x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc3(x, adj)
        return x


def demo():
    feature = torch.ones(size=(5, 9, 20)).cuda()  # batch_size, node, feature_dim
    adj = torch.ones(size=(9, 9)).cuda()
    gcn = GCN(inputDim=20, hidDim=10, outputDim=20, dropout=0.5).cuda()
    x = gcn(feature, adj)
    print(x.shape)


# demo()
# 我可以看吗？hhhhh就是叫你先看
