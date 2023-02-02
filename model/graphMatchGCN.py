import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from GCN.layer import GraphConvolution


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


# class GraphConvolution(nn.Module):
#     """
#     Document : https://github.com/tkipf/pygcn
#                https://github.com/bamos/block
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         """
#         参数解释
#         (构造第一层GCN的时候)
#         in_features:  64
#         out_features: 128
#
#         (构造第二层GCN的时候)
#         in_features:  64
#         out_features: 128
#         """
#
#         super(GraphConvolution, self).__init__()
#
#         self.in_features = in_features
#         self.out_features = out_features
#
#         self.FC = nn.Linear(in_features, out_features, bias=bias)  # ! 只定义了一个全连接层？
#         self.FC.apply(init_weights)  # 初始化
#
#     def forward(self, input, adj):
#         """
#         参数解释
#         input：对图像提取特征之后的一个tensor，shape是batch * 6 * 64
#         adj  ：6 * 6经过初始化后的矩阵
#         """
#
#         batchSize, numOfNode = input.size(0), adj.size(0)
#         support = self.FC(input)  # support的shape(batch * 6 * 128)
#         # print(f"support = {support.shape}")
#         Adj = adj.repeat(batchSize, 1, 1)  # 代表只拓展第一维
#         output = torch.bmm(Adj, support)  # 将Adj和support矩阵相乘，output的shape（batch，128），这就是paper说的用图像初始化邻接矩阵？
#
#         return output
#
#         # In order to support batch operation, we should design a funtion like scipy.linalg.block_diag to bulid Adj Matrix.
#         # Adj Matrix should be [batchSize * numNode, batchSize * numNode]
#
#         # Abandoned Code 1:
#         # Adj = torch.zeros((numOfNode*batchSize, numOfNode*batchSize),
#         #                   dtype=torch.float,
#         #                   requires_grad=True).cuda() if input.is_cuda else torch.zeros((numOfNode*batchSize, numOfNode*batchSize),
#         #                                                                                dtype=torch.float,
#         #                                                                                requires_grad=True)
#         # for index in range(batchSize):
#         #     Adj[index*numOfNode:(index+1)*numOfNode, index*numOfNode:(index+1)*numOfNode] = adj
#         # output = torch.mm(Adj, support.view(batchSize*numOfNode, -1))
#         # output = output.view(batchSize, numOfNode, -1)
#
#         # Abandoned Code 2:
#         # Adj = block.block_diag([adj.cpu() for i in range(batchSize)])
#         # if input.is_cuda:
#         #     Adj = Adj.cuda()
#         # output = torch.mm(Adj, support.view(batchSize*numOfNode, -1))
#         # output = output.view(batchSize, numOfNode, -1)


# class GCN(nn.Module):
#     def __init__(self, dim_in, dim_hid, dim_out):
#         super(GCN, self).__init__()
#         self.gcn_1 = GraphConvolution(dim_in, dim_hid)
#         self.gcn_2 = GraphConvolution(dim_hid, dim_out)
#         self.dropout = nn.Dropout()
#         self.relu = nn.ReLU()
#
#     def forward(self, x, adj):
#         x = self.gcn_1(x, adj)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.gcn_2(x, adj)
#         return x


class GCNwithIntraAndInterMatrix(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out,
                 useRandomMatrix=False, useAllOneMatrix=False):
        """
        参数解释
        dim_in : 2048
        dim_hid: 1024
        dim_out: 2048
        useIntroGCN: true
        useInterGCN: true
        useRandomMatrix: false
        useAllOneMatrix: false
        """

        super(GCNwithIntraAndInterMatrix, self).__init__()

        self.useRandomMatrix = useRandomMatrix
        self.useAllOneMatrix = useAllOneMatrix

        # 定义了两层GCN
        # todo without GCN1
        self.IntraGCN_1 = GraphConvolution(dim_in, dim_hid)
        self.IntraGCN_2 = GraphConvolution(dim_hid, dim_out)
        # todo 2022/2/13 18:35
        # todo without GCN2
        self.InterGCN_1 = GraphConvolution(dim_in, dim_hid)  
        self.InterGCN_2 = GraphConvolution(dim_hid, dim_out)
        self.Dropout = nn.Dropout()
        self.ReLU = nn.ReLU()

        # Inter Adjacency Matrix
        if self.useRandomMatrix:
            InterAdjMatrix = torch.rand((20, 20), dtype=torch.float)
        if self.useAllOneMatrix:
            InterAdjMatrix = torch.ones((20, 20), dtype=torch.float)

            # todo 2022/2/13 18:35
        InterAdjMatrix[0:10, 0:10] = 0.0
        InterAdjMatrix[10:, 10:] = 0.0
        #
        #     # Self Link
        for i in range(20):
            InterAdjMatrix[i, i] = 1.0
        InterAdjMatrix.requires_grad = True
        self.InterAdjMatrix = nn.Parameter(InterAdjMatrix, requires_grad=True)

    def forward(self, SourceFeature, IntraAdjMatrix, MomFeature=None, momIntraAdjMatrix=None, is_Mom=False,
                is_proto=False, is_Nproto=False, intra_graph=None):  # Size of feature : Batch * 12 * 64
        # print(self.InterAdjMatrix)
        # Update Intra/Inter Adj Matrix
        # self.InterAdjMatrix.data.copy_(self.InterAdjMatrix * self.InterMaskMatrix)
        """
        A.copy_（B*C）这里有两个key points
        B * C: 就是将B和C矩阵中的每个对应位置的值直接相乘，因此B和C的shape必须完全一样
        copy_: 就是将B*C的结果复制到A中
        """
        if IntraAdjMatrix is not None:
            IntraAdjMatrix.data.clamp_(min=0)  # clamp_的作用是将tensor中小于min的值变为min
            IntraAdjMatrix.data.copy_(IntraAdjMatrix / IntraAdjMatrix.sum(dim=1, keepdim=True))
        if momIntraAdjMatrix is not None:
            momIntraAdjMatrix.data.clamp_(min=0)  # clamp_的作用是将tensor中小于min的值变为min
            momIntraAdjMatrix.data.copy_(momIntraAdjMatrix / momIntraAdjMatrix.sum(dim=1, keepdim=True))
        nodeNum = SourceFeature.shape[1]
        # print(nodeNum)
        # Intra GCN
        if is_proto:
            # print("is_proto-GCN")
            self.InterAdjMatrix.data.clamp_(min=0)
            self.InterAdjMatrix.data.copy_(self.InterAdjMatrix / self.InterAdjMatrix.sum(dim=1, keepdim=True))
            # todo without GCN1
            if not is_Nproto and intra_graph is None:
                SourceFeature = self.IntraGCN_1(SourceFeature, IntraAdjMatrix)  # Batch * 10 * 2048
                SourceFeature = self.Dropout(self.ReLU(SourceFeature))  # Batch * 10 * 1024
                SourceFeature = self.IntraGCN_2(SourceFeature, IntraAdjMatrix)  # Batch * 10 * 2048
            else:
                SourceFeature = intra_graph

            MomFeature = self.IntraGCN_1(MomFeature, momIntraAdjMatrix)  # Batch * 10 * 1024
            MomFeature = self.Dropout(self.ReLU(MomFeature))  # Batch * 10 * 1024
            MomFeature = self.IntraGCN_2(MomFeature, momIntraAdjMatrix)  # Batch * 10 * 2048

            # Concat Source/Target Feature 重新汇集两个特征图
            Feature = torch.cat((SourceFeature, MomFeature), 1)  # Batch * 20 * 2048
            # todo without GCN2
            # Inter GCN
            Feature = self.InterGCN_1(Feature, self.InterAdjMatrix)  # Batch * 20 * 1024
            Feature = self.Dropout(self.ReLU(Feature))
            Feature = self.InterGCN_2(Feature, self.InterAdjMatrix)  # Batch * 20 * 2048

            return SourceFeature, Feature[:, 0:nodeNum, :], Feature[:, nodeNum:, :]

        else:
            # todo without GCN1
            SourceFeature = self.IntraGCN_1(SourceFeature, IntraAdjMatrix)  # Batch * 10 * 1024
            SourceFeature = self.Dropout(self.ReLU(SourceFeature))  # Batch * 10 * 1024
            SourceFeature = self.IntraGCN_2(SourceFeature, IntraAdjMatrix)  # Batch * 10 * 2048

            return SourceFeature
