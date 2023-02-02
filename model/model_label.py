import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50
from utils.utlits import get_region, normalize_adj_torch, randomint_plus
from graphMatchGCN import GCNwithIntraAndInterMatrix


class MoProGCN(nn.Module):
    def __init__(self, inputDim, nodeNum, num_classes=1000):
        super(MoProGCN, self).__init__()
        self.nodeNum = nodeNum
        self.inputDim = inputDim
        self.num_classes = num_classes
        # encoder
        self.encoder_q = resnet50(pretrained=True)  # resnet50

        self.gcn_q = GCNwithIntraAndInterMatrix(dim_in=2048, dim_hid=1024, dim_out=2048, useAllOneMatrix=True)
        # momentum encoder,其参数从encoder中复制得到,采用动量模式利用encoder的参数来更新，见第 32 行
        # self.encoder_k = resnet50(pretrained=False)  # resnet50--fix
        # self.gcn_k = GCNwithIntraAndInterMatrix(dim_in=2048, dim_hid=1024, dim_out=2048, useAllOneMatrix=True)
        self.MLP_norm_q = nn.Sequential(
            nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim),
            nn.ReLU(),
            nn.Linear(in_features=self.inputDim, out_features=1024),
            Normalize(2))
        self.MLP_norm_k = nn.Sequential(
            nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim),
            nn.ReLU(),
            nn.Linear(in_features=self.inputDim, out_features=1024),
            Normalize(2))

        for param_q, param_k in zip(self.MLP_norm_q.parameters(), self.MLP_norm_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.avg = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_similarity = nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim)
        self.fc_g = nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim)
        self.fc_cls = nn.Linear(in_features=self.inputDim, out_features=self.num_classes)

        self.adjMatrix_q = torch.nn.Parameter(normalize_adj_torch(torch.ones(size=(self.nodeNum, self.nodeNum))),
                                              requires_grad=True)
        self.adjMatrix_k = torch.nn.Parameter(normalize_adj_torch(torch.ones(size=(self.nodeNum, self.nodeNum))),
                                              requires_grad=False)

        self.register_buffer("prototypes",
                             torch.zeros(num_classes, self.nodeNum, self.inputDim))  # (num_class, 10, 128)
        #self.prototypes=torch.nn.Parameter(torch.zeros(num_classes, self.nodeNum, self.inputDim),requires_grad=False)


    # @torch.no_grad()
    # todo momentum encoder参数更新
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        with torch.no_grad():
            for param_q, param_k in zip(self.MLP_norm_q.parameters(), self.MLP_norm_k.parameters()):
                param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
            self.adjMatrix_k.data = self.adjMatrix_k.data * args.moco_m + self.adjMatrix_q.data * (1. - args.moco_m)

    def forward(self, batch, args, is_eval=False, is_proto=False, is_clean=False, epoch=0):
        
        img = batch[0]
        target = batch[1]
        batchsize = img.shape[0]
        similarity_prototypes = torch.zeros(1).cuda()
        N_similarity_prototypes = torch.zeros(1).cuda()
        if is_proto and not is_eval:
            N_target = randomint_plus(0, self.num_classes, cutoff=target, size=(1, batchsize))
            N_target = torch.from_numpy(N_target).cuda()
            prototypes = self.prototypes[target].clone()
            # same class
            pred, Feature, nodeFeature, ProFeature = self.Res_Gcn(self.encoder_q, self.gcn_q,
                                                                  img, self.avg, self.fc_cls, self.fc_g, is_proto=True,
                                                                  prototypes=prototypes, adj_q=self.adjMatrix_q,
                                                                  adj_k=self.adjMatrix_k)
            similarity_prototypes = torch.mean((torch.cosine_similarity(nodeFeature, ProFeature, dim=1)),
                                               dim=0)
            # different class
            for i in range(1):
                N_prototypes = self.prototypes[N_target[i]].clone()
                N_pred, N_Feature, N_nodeFeature, N_ProFeature = self.Res_Gcn(self.encoder_q, self.gcn_q,
                                                                              img, self.avg, self.fc_cls, self.fc_g,
                                                                              is_proto=True,
                                                                              prototypes=N_prototypes,
                                                                              adj_q=self.adjMatrix_q,
                                                                              adj_k=self.adjMatrix_k)
                N_similarity_prototypes = torch.mean(torch.cosine_similarity(N_nodeFeature, N_ProFeature, dim=1),
                                                     dim=0)

            # N_similarity_prototypes /= 1
            # logits proto
            prototypes = self.prototypes.clone().detach()
            prototypes = prototypes.view(-1, self.nodeNum * self.inputDim)
            cls_Feature = Feature.view(batchsize, self.nodeNum * self.inputDim)

            prototypes = self.MLP_norm_q(prototypes)
            cls_Feature = self.MLP_norm_k(cls_Feature)
            logits_proto = torch.mm(cls_Feature, prototypes.t()) / args.temperature
        else:
            pred, Feature = self.Res_Gcn(self.encoder_q, self.gcn_q,
                                         img, self.avg, self.fc_cls,
                                         self.fc_g,
                                         adj_q=self.adjMatrix_q)  # node_feature 为10个结点维度为2048的graph,output为graph经过concat后的特征表示
            logits_proto = torch.zeros(size=(batchsize, self.num_classes)).cuda()
        if is_eval:
            return pred
            # return pred, Feature
        # 动量网络
        with torch.no_grad():
            self._momentum_update_key_encoder(args)  # update the momentum encoder

        if is_clean:
            # logits_proto = F.softmax(logits_proto, dim=1)
            #伪得分
            soft_label = args.alpha * F.softmax(pred, dim=1) + (1 - args.alpha) * F.softmax(logits_proto, dim=1)

            # keep ground truth label
            gt_score = soft_label[target >= 0, target]  # 原标签得分
            clean_idx = gt_score > (1 / args.num_class)  # 原标签是否大于 1/K
            # assign a new pseudo label
            max_score, hard_label = soft_label.max(1)
            correct_idx = max_score > args.pseudo_th
            target[correct_idx] = hard_label[correct_idx]
            clean_idx = clean_idx | correct_idx
            # print(clean_idx)
        if is_clean:
            clean_idx = clean_idx.bool()
            # update momentum prototypes with pseudo-labels
            with torch.no_grad():
                for feat, label in zip(Feature[clean_idx], target[clean_idx]):
                    self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
            target = target[clean_idx]
            pred = pred[clean_idx]
            logits_proto = logits_proto[clean_idx]
            # pred_mom = pred_mom[clean_idx]
        else:
            with torch.no_grad():
                for feat, label in zip(Feature, target):
                    self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat

        self.prototypes = F.normalize(self.prototypes.detach(), p=2, dim=1)
        
        return pred, logits_proto, target, similarity_prototypes, N_similarity_prototypes
        # only res+gcn
        # return pred, None, None, similarity_net, similarity_prototypes, N_similarity_prototypes

    def Res_Gcn(self, encoder, gcn, input, avg, fc_cls, fc_g, MomEmbedding=False, adj_q=None, adj_k=None,
                is_proto=False, prototypes=None,
                Feature=None):
        batchsize = input.shape[0]
        backbone_feature = encoder(input)  # output of resnet50 shape(batchsize, inputDim, 14, 14)
        local_feature = avg(backbone_feature).view(batchsize, -1)  # (batchsize, inputDim)
        region_feature = get_region(backbone_feature)  # (batchsize, nodeNum, inputDim)
        region_feature = torch.cat((region_feature, local_feature.unsqueeze(1)), dim=1)
        # return batchsize, backbone_feature
        if is_proto:
            feature, node_feature, momNode_feature = gcn(SourceFeature=region_feature, IntraAdjMatrix=adj_q,
                                                         MomFeature=prototypes, momIntraAdjMatrix=adj_k,
                                                         is_proto=True)
            global_feature = fc_g(feature.view(batchsize, self.nodeNum * self.inputDim))
            output = fc_cls(global_feature)
            Feature = self.fc_similarity(node_feature.view(batchsize, self.nodeNum * self.inputDim))
            ProFeature = self.fc_similarity(momNode_feature.view(batchsize, self.nodeNum * self.inputDim))

            return output, region_feature, Feature, ProFeature
        else:
            node_feature = gcn(region_feature, adj_q)
            global_feature = node_feature.view(batchsize, self.nodeNum * self.inputDim)  # (batchsize, nodeNum*inputDim)
            global_feature = fc_g(global_feature)  # (batchsize, inputDim))
            output = fc_cls(global_feature)
            # todo feature heatmap
            # return output, backbone_feature
            return output, region_feature

# todo 对齐所有类别
# class MoProGCN(nn.Module):
#     def __init__(self, inputDim, nodeNum, num_classes=1000):
#         super(MoProGCN, self).__init__()
#         self.nodeNum = nodeNum
#         self.inputDim = inputDim
#         self.num_classes = num_classes
#         # encoder
#         self.encoder_q = resnet50(pretrained=True)  # resnet50

#         self.gcn_q = GCNwithIntraAndInterMatrix(dim_in=2048, dim_hid=1024, dim_out=2048, useAllOneMatrix=True)

#         self.MLP_norm_q = nn.Sequential(
#             nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim),
#             nn.ReLU(),
#             nn.Linear(in_features=self.inputDim, out_features=1024),
#             Normalize(2))

#         # for param_q, param_k in zip(self.MLP_norm_q.parameters(), self.MLP_norm_k.parameters()):
#         #     param_k.data.copy_(param_q.data)  # initialize
#         #     param_k.requires_grad = False  # not update by gradient

#         self.avg = nn.AdaptiveAvgPool2d(output_size=1)
#         self.fc_similarity = nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim)
#         self.fc_g = nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim)
#         self.fc_cls = nn.Linear(in_features=self.inputDim, out_features=self.num_classes)

#         self.adjMatrix_q = torch.nn.Parameter(normalize_adj_torch(torch.ones(size=(self.nodeNum, self.nodeNum))),
#                                               requires_grad=True)
#         self.adjMatrix_k = torch.nn.Parameter(normalize_adj_torch(torch.ones(size=(self.nodeNum, self.nodeNum))),
#                                               requires_grad=False)

#         self.register_buffer("prototypes",
#                              torch.zeros(num_classes, self.nodeNum, self.inputDim))  # (num_class, 10, 128)
#         #self.prototypes=torch.nn.Parameter(torch.zeros(num_classes, self.nodeNum, self.inputDim),requires_grad=False)


#     @torch.no_grad()
#     # todo momentum encoder参数更新
#     def _momentum_update_key_encoder(self, args):
#         """
#         update momentum encoder
#         """
#         with torch.no_grad():
#             self.adjMatrix_k.data = self.adjMatrix_k.data * args.moco_m + self.adjMatrix_q.data * (1. - args.moco_m)

#     def forward(self, batch, args, is_eval=False, is_proto=False, is_clean=False, epoch=0):
        
#         img = batch[0]
#         target = batch[1]
#         batchsize = img.shape[0]
#         similarity_prototypes = torch.zeros(1).cuda()
#         N_similarity_prototypes = torch.zeros(1).cuda()
#         similarity_score = torch.zeros(size=(batchsize, self.num_classes), dtype=torch.float).cuda()
#         backbone_feature = self.encoder_q(img)  # output of resnet50 shape(batchsize, inputDim, 14, 14)
#         global_feature = self.avg(backbone_feature).view(batchsize, -1)  # (batchsize, inputDim)
#         local_features = get_region(backbone_feature)  # (batchsize, nodeNum, inputDim)
#         instance_graph = torch.cat((local_features, global_feature.unsqueeze(1)), dim=1)

#         if is_proto and not is_eval:
#             class_array = np.arange(self.num_classes)
#             N_target = np.zeros(shape=(self.num_classes-1, batchsize))
#             for index in range(N_target.shape[1]):
#                 instance_target = target[index]
#                 N_target[:,index] = np.delete(class_array, instance_target.item())
#             # N_target = torch.from_numpy(N_target).cuda()
#             prototypes = self.prototypes[target].clone()
#             # same class
#             intra_graph, node_feature, momNode_feature = self.gcn_q(SourceFeature=instance_graph, IntraAdjMatrix=self.adjMatrix_q,
#                                                          MomFeature=prototypes, momIntraAdjMatrix=self.adjMatrix_k,
#                                                          is_proto=True)
#             cls_feature = self.fc_g(node_feature.view(batchsize, self.nodeNum * self.inputDim))
#             pred = self.fc_cls(cls_feature)
#             InsFeature = self.fc_similarity(node_feature.view(batchsize, self.nodeNum * self.inputDim))
#             ProFeature = self.fc_similarity(momNode_feature.view(batchsize, self.nodeNum * self.inputDim))
#             cos_score = torch.cosine_similarity(InsFeature, ProFeature, dim=1)
#             # store similarity score for each instance
#             similarity_score[:, target] = cos_score
#             ##
#             similarity_prototypes = torch.mean(cos_score, dim=0)
#             # different class
#             for i in range(self.num_classes-1):
#                 N_prototypes = self.prototypes[N_target[i]].clone()
#                 _, N_nodeFeature, N_ProFeature = self.gcn_q(SourceFeature=instance_graph, IntraAdjMatrix=self.adjMatrix_q,
#                                                          MomFeature=N_prototypes, momIntraAdjMatrix=self.adjMatrix_k,
#                                                          is_proto=True, is_Nproto=True, intra_graph=intra_graph)
#                 N_InsFeature = self.fc_similarity(N_nodeFeature.view(batchsize, self.nodeNum * self.inputDim))
#                 N_ProFeature = self.fc_similarity(N_ProFeature.view(batchsize, self.nodeNum * self.inputDim))
#                 N_cos_score = torch.cosine_similarity(N_InsFeature, N_ProFeature, dim=1)
#                 similarity_score[:, N_target[i]] = N_cos_score
#                 N_similarity_prototypes += torch.mean(N_cos_score, dim=0)

#             N_similarity_prototypes /= self.num_classes - 1
#             # logits proto
#             prototypes = self.prototypes.clone().detach()
#             prototypes = prototypes.view(-1, self.nodeNum * self.inputDim)
#             cls_Feature = instance_graph.view(batchsize, self.nodeNum * self.inputDim)

#             prototypes = self.MLP_norm_q(prototypes)
#             cls_Feature = self.MLP_norm_q(cls_Feature)
#             logits_proto = torch.mm(cls_Feature, prototypes.t()) / args.temperature
#         else:
#             node_feature = self.gcn_q(SourceFeature=instance_graph, IntraAdjMatrix=self.adjMatrix_q)
#             cls_feature = self.fc_g(node_feature.view(batchsize, self.nodeNum * self.inputDim))
#             pred = self.fc_cls(cls_feature)
#             logits_proto = torch.zeros(size=(batchsize, self.num_classes)).cuda()
#         if is_eval:
#             return pred
#             # return pred, Feature
#         # 动量网络
#         with torch.no_grad():
#             self._momentum_update_key_encoder(args)  # update the momentum encoder

#         if is_clean:
#             logits_proto = F.softmax(logits_proto, dim=1)
#             #伪得分
#             soft_label = args.alpha * F.softmax(pred, dim=1) + (1 - args.alpha) * F.softmax(similarity_score, dim=1)

#             # keep ground truth label
#             gt_score = soft_label[target >= 0, target]  # 原标签得分
#             clean_idx = gt_score > (1 / args.num_class)  # 原标签是否大于 1/K
#             # assign a new pseudo label
#             max_score, hard_label = soft_label.max(1)
#             correct_idx = max_score > args.pseudo_th
#             target[correct_idx] = hard_label[correct_idx]
#             clean_idx = clean_idx | correct_idx
#             # print(clean_idx)
#         if is_clean:
#             clean_idx = clean_idx.bool()
#             # update momentum prototypes with pseudo-labels
#             with torch.no_grad():
#                 for feat, label in zip(instance_graph[clean_idx], target[clean_idx]):
#                     self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
#             target = target[clean_idx]
#             pred = pred[clean_idx]
#             logits_proto = logits_proto[clean_idx]

#         else:
#             with torch.no_grad():
#                 for feat, label in zip(instance_graph, target):
#                     self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat

#         self.prototypes = F.normalize(self.prototypes.detach(), p=2, dim=1)
        
#         return pred, logits_proto, target, similarity_prototypes, N_similarity_prototypes
        # only res+gcn
        # return pred, None, None, similarity_net, similarity_prototypes, N_similarity_prototypes

    # def Res_Gcn(self, encoder, gcn, input, avg, fc_cls, fc_g, MomEmbedding=False, adj_q=None, adj_k=None,
    #             is_proto=False, prototypes=None,
    #             Feature=None):
    #     batchsize = input.shape[0]
    #     backbone_feature = encoder(input)  # output of resnet50 shape(batchsize, inputDim, 14, 14)
    #     local_feature = avg(backbone_feature).view(batchsize, -1)  # (batchsize, inputDim)
    #     region_feature = get_region(backbone_feature)  # (batchsize, nodeNum, inputDim)
    #     region_feature = torch.cat((region_feature, local_feature.unsqueeze(1)), dim=1)
    #     # return batchsize, backbone_feature
    #     if is_proto:
    #         feature, node_feature, momNode_feature = gcn(SourceFeature=region_feature, IntraAdjMatrix=adj_q,
    #                                                      MomFeature=prototypes, momIntraAdjMatrix=adj_k,
    #                                                      is_proto=True)
    #         global_feature = fc_g(feature.view(batchsize, self.nodeNum * self.inputDim))
    #         output = fc_cls(global_feature)
    #         Feature = self.fc_similarity(node_feature.view(batchsize, self.nodeNum * self.inputDim))
    #         ProFeature = self.fc_similarity(momNode_feature.view(batchsize, self.nodeNum * self.inputDim))

    #         return output, region_feature, Feature, ProFeature
    #     else:
    #         node_feature = gcn(region_feature, adj_q)
    #         global_feature = node_feature.view(batchsize, self.nodeNum * self.inputDim)  # (batchsize, nodeNum*inputDim)
    #         global_feature = fc_g(global_feature)  # (batchsize, inputDim))
    #         output = fc_cls(global_feature)
    #         # todo feature heatmap
    #         # return output, backbone_feature
    #         return output, region_feature


# todo for BCNN
# class MoProGCN(nn.Module):
#     def __init__(self, inputDim, nodeNum, num_classes=1000):
#         super(MoProGCN, self).__init__()
#         self.nodeNum = nodeNum
#         self.inputDim = inputDim
#         self.num_classes = num_classes
#         # encoder
#         self.encoder_q = BCNN(pretrained=True)  # resnet50

#         self.gcn_q = GCNwithIntraAndInterMatrix(dim_in=512, dim_hid=256, dim_out=512, useAllOneMatrix=True)
#         # momentum encoder,其参数从encoder中复制得到,采用动量模式利用encoder的参数来更新，见第 32 行
#         # self.encoder_k = resnet50(pretrained=False)  # resnet50--fix
#         # self.gcn_k = GCNwithIntraAndInterMatrix(dim_in=2048, dim_hid=1024, dim_out=2048, useAllOneMatrix=True)
#         self.MLP_norm_q = nn.Sequential(
#             nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim),
#             nn.ReLU(),
#             nn.Linear(in_features=self.inputDim, out_features=256),
#             Normalize(2))
#         self.MLP_norm_k = nn.Sequential(
#             nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim),
#             nn.ReLU(),
#             nn.Linear(in_features=self.inputDim, out_features=256),
#             Normalize(2))

#         for param_q, param_k in zip(self.MLP_norm_q.parameters(), self.MLP_norm_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient

#         self.avg = nn.AdaptiveAvgPool2d(output_size=1)
#         self.fc_similarity = nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim)
#         self.fc_g = nn.Linear(in_features=self.nodeNum * self.inputDim, out_features=self.inputDim)
#         self.fc_cls = nn.Linear(in_features=self.inputDim, out_features=self.num_classes)

#         self.adjMatrix_q = torch.nn.Parameter(normalize_adj_torch(torch.ones(size=(self.nodeNum, self.nodeNum))),
#                                               requires_grad=True)
#         self.adjMatrix_k = torch.nn.Parameter(normalize_adj_torch(torch.ones(size=(self.nodeNum, self.nodeNum))),
#                                               requires_grad=False)

#         self.register_buffer("prototypes",
#                              torch.zeros(num_classes, self.nodeNum, self.inputDim))  # (num_class, 10, 128)
#         #self.prototypes=torch.nn.Parameter(torch.zeros(num_classes, self.nodeNum, self.inputDim),requires_grad=False)


#     # @torch.no_grad()
#     # todo momentum encoder参数更新
#     def _momentum_update_key_encoder(self, args):
#         """
#         update momentum encoder
#         """
#         with torch.no_grad():
#             for param_q, param_k in zip(self.MLP_norm_q.parameters(), self.MLP_norm_k.parameters()):
#                 param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)
#             # todo 修改 2022/2/22 20:28
#             self.adjMatrix_k.data = self.adjMatrix_k.data * args.moco_m + self.adjMatrix_q.data * (1. - args.moco_m)

#     def forward(self, batch, args, is_eval=False, is_proto=False, is_clean=False, epoch=0):
        
#         img = batch[0]
#         img_aug = batch[1]  # 数据增强强化版，用于momentum encoder
#         target = batch[2]
#         batchsize = img.shape[0]
#         similarity_net = torch.zeros(1).cuda()
#         similarity_prototypes = torch.zeros(1).cuda()
#         N_similarity_prototypes = torch.zeros(1).cuda()
#         if is_proto and not is_eval:
#             N_target = randomint_plus(0, self.num_classes, cutoff=target, size=(1, batchsize))
#             N_target = torch.from_numpy(N_target).cuda()
#             prototypes = self.prototypes[target].clone()
#             # same class
#             pred, Feature, nodeFeature, ProFeature = self.Res_Gcn(self.encoder_q, self.gcn_q,
#                                                                   img, self.avg, self.fc_cls, self.fc_g, is_proto=True,
#                                                                   prototypes=prototypes, adj_q=self.adjMatrix_q,
#                                                                   adj_k=self.adjMatrix_k)
#             similarity_prototypes = torch.mean((torch.cosine_similarity(nodeFeature, ProFeature, dim=1)),
#                                                dim=0)
#             # different class
#             for i in range(1):
#                 N_prototypes = self.prototypes[N_target[i]].clone()
#                 N_pred, N_Feature, N_nodeFeature, N_ProFeature = self.Res_Gcn(self.encoder_q, self.gcn_q,
#                                                                               img, self.avg, self.fc_cls, self.fc_g,
#                                                                               is_proto=True,
#                                                                               prototypes=N_prototypes,
#                                                                               adj_q=self.adjMatrix_q,
#                                                                               adj_k=self.adjMatrix_k)
#                 N_similarity_prototypes = torch.mean(torch.cosine_similarity(N_nodeFeature, N_ProFeature, dim=1),
#                                                      dim=0)

#             # N_similarity_prototypes /= 1
#             # logits proto
#             prototypes = self.prototypes.clone().detach()
#             prototypes = prototypes.view(-1, self.nodeNum * self.inputDim)
#             cls_Feature = Feature.view(batchsize, self.nodeNum * self.inputDim)

#             prototypes = self.MLP_norm_q(prototypes)
#             cls_Feature = self.MLP_norm_k(cls_Feature)
#             logits_proto = torch.mm(cls_Feature, prototypes.t()) / args.temperature
#         else:
#             pred, Feature = self.Res_Gcn(self.encoder_q, self.gcn_q,
#                                          img, self.avg, self.fc_cls,
#                                          self.fc_g,
#                                          adj_q=self.adjMatrix_q)  # node_feature 为10个结点维度为2048的graph,output为graph经过concat后的特征表示
#             logits_proto = torch.zeros(size=(batchsize, self.num_classes)).cuda()
#         if is_eval:
#             return pred
#             # return pred, Feature
#         # 动量网络
#         with torch.no_grad():
#             self._momentum_update_key_encoder(args)  # update the momentum encoder

#         if is_clean:
#             # logits_proto = F.softmax(logits_proto, dim=1)
#             #伪得分
#             soft_label = args.alpha * F.softmax(pred, dim=1) + (1 - args.alpha) * F.softmax(logits_proto, dim=1)

#             # keep ground truth label
#             gt_score = soft_label[target >= 0, target]  # 原标签得分
#             clean_idx = gt_score > (1 / args.num_class)  # 原标签是否大于 1/K
#             # assign a new pseudo label
#             max_score, hard_label = soft_label.max(1)
#             correct_idx = max_score > args.pseudo_th
#             target[correct_idx] = hard_label[correct_idx]
#             clean_idx = clean_idx | correct_idx
#             # print(clean_idx)
#         if is_clean:
#             clean_idx = clean_idx.bool()
#             # update momentum prototypes with pseudo-labels
#             with torch.no_grad():
#                 for feat, label in zip(Feature[clean_idx], target[clean_idx]):
#                     self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat
#             target = target[clean_idx]
#             pred = pred[clean_idx]
#             logits_proto = logits_proto[clean_idx]
#             # pred_mom = pred_mom[clean_idx]
#         else:
#             with torch.no_grad():
#                 for feat, label in zip(Feature, target):
#                     self.prototypes[label] = self.prototypes[label] * args.proto_m + (1 - args.proto_m) * feat

#         self.prototypes = F.normalize(self.prototypes.detach(), p=2, dim=1)
        
#         return pred, logits_proto, target, similarity_prototypes, N_similarity_prototypes
#         # only res+gcn
#         # return pred, None, None, similarity_net, similarity_prototypes, N_similarity_prototypes

#     def Res_Gcn(self, encoder, gcn, input, avg, fc_cls, fc_g, MomEmbedding=False, adj_q=None, adj_k=None,
#                 is_proto=False, prototypes=None,
#                 Feature=None):
#         batchsize = input.shape[0]
#         backbone_feature = encoder(input)  # output of resnet50 shape(batchsize, inputDim, 14, 14)
#         local_feature = avg(backbone_feature).view(batchsize, -1)  # (batchsize, inputDim)
#         region_feature = get_region(backbone_feature, fixed_size=(backbone_feature.shape[2], backbone_feature.shape[3]))  # (batchsize, nodeNum, inputDim)
#         region_feature = torch.cat((region_feature, local_feature.unsqueeze(1)), dim=1)
#         # return batchsize, backbone_feature
#         if is_proto:
#             feature, node_feature, momNode_feature = gcn(SourceFeature=region_feature, IntraAdjMatrix=adj_q,
#                                                          MomFeature=prototypes, momIntraAdjMatrix=adj_k,
#                                                          is_proto=True)
#             global_feature = fc_g(node_feature.view(batchsize, self.nodeNum * self.inputDim))
#             output = fc_cls(global_feature)
#             Feature = self.fc_similarity(node_feature.view(batchsize, self.nodeNum * self.inputDim))
#             ProFeature = self.fc_similarity(momNode_feature.view(batchsize, self.nodeNum * self.inputDim))

#             return output, region_feature, Feature, ProFeature
#         else:
#             node_feature = gcn(region_feature, adj_q)
#             global_feature = node_feature.view(batchsize, self.nodeNum * self.inputDim)  # (batchsize, nodeNum*inputDim)
#             global_feature = fc_g(global_feature)  # (batchsize, inputDim))
#             output = fc_cls(global_feature)
#             # todo feature heatmap
#             # return output, backbone_feature
#             return output, region_feature





class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
