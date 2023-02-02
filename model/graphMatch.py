import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self, feature_dim, hidden_sizes=None, softmax=False):
        """Constructor.

        Args:
          hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outputs.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(MLP, self).__init__()

        # this also handles the case of an empty list
        super().__init__()
        self.softmax = softmax
        self._feature_dim = feature_dim
        # self._edge_feature_dim = edge_feature_dim
        self._hidden_sizes = hidden_sizes if hidden_sizes else None
        # self._edge_hidden_sizes = edge_hidden_sizes
        self._build_model()

    def _build_model(self):
        layers = []
        layers.append(nn.Linear(self._feature_dim, self._hidden_sizes[0]))
        for i in range(1, len(self._hidden_sizes)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(self._hidden_sizes[i - 1], self._hidden_sizes[i]))
        for layer in layers:
            if isinstance(layers, nn.Linear):
                nn.init.kaiming_normal_(layer.weight.data)
                # layer.weight.data *= 0.1
        self.MLP1 = nn.Sequential(*layers)

    def forward(self, node_features, edge_features=None):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self._hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = self.MLP1(node_features)
        # if edge_features is None or self._edge_hidden_sizes is None:
        #     edge_outputs = edge_features
        # else:
        #     edge_outputs = self.MLP2(edge_features)
        if self.softmax:
            node_outputs = torch.softmax(node_outputs, dim=0)
        return node_outputs  # , edge_outputs


def graph_Propagation_layer(node, edg, layer):
    node_intra = torch.zeros_like(node)
    for index_m, feature_m in enumerate(node):
        for index_n, feature_n in enumerate(node):
            node_intra[index_m] += layer(torch.cat((feature_m, feature_n, edg[index_m][index_n]), dim=0))
        node_intra[index_m] /= node.shape[0]

    return node_intra


def graph_interaction_layer(graph_1_node, graph_2_node):
    matching_weights_1 = torch.zeros_like(graph_1_node)
    matching_weights_2 = torch.zeros_like(graph_2_node)
    for index, node in enumerate(graph_1_node):
        matching_weights_1[index] = torch.mean(graph_2_node * (torch.exp(node * graph_2_node)
                                              / torch.sum(node * graph_2_node, dim=0)), dim=0)
    for index, node in enumerate(graph_2_node):
        matching_weights_2[index] = torch.mean(graph_1_node * (torch.exp(node * graph_1_node)
                                              / torch.sum(node * graph_1_node, dim=0)), dim=0)

    return matching_weights_1, matching_weights_2


def one_hot(dim, index, cuda=False):
    if cuda:
        result = torch.zeros(dim, dtype=torch.float).cuda()
    else:
        result = torch.zeros(dim, dtype=torch.float)
    result[index] = 1.0

    return result


def create_edg(node_feature):
    node_num = node_feature.shape[0]
    edg_feature = []
    for i in range(node_num):
        from_one_hot = one_hot(dim=node_num, index=i, cuda=True)
        for j in range(node_num):
            to_one_hot = one_hot(dim=node_num, index=j, cuda=True)
            edg_feature.append(torch.cat((node_feature[i], from_one_hot,
                                          node_feature[j], to_one_hot)))

    edg_feature = torch.stack(edg_feature, dim=0)
    return edg_feature


def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(r_mat_inv_sqrt, mx)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)

    return mx


class GraphMatch(nn.Module):
    def __init__(self, node_num, node_feature_dim):
        super(GraphMatch, self).__init__()
        self.node_num = node_num
        self.node_intra_dim = node_feature_dim
        self.node_cross_dim = node_feature_dim

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = node_feature_dim
        self.Node_embedding = MLP(feature_dim=node_feature_dim, hidden_sizes=[512, node_feature_dim])
        self.edge_embedding = MLP(feature_dim=(node_feature_dim*2+node_num*2), hidden_sizes=[512, node_feature_dim])

        self.graph_propagation_layer = MLP(feature_dim=(self.node_feature_dim*2 + self.edge_feature_dim),
                                           hidden_sizes=[512, node_feature_dim])
        self.graph_update_layer = MLP(
            feature_dim=(self.node_feature_dim + self.node_intra_dim + self.node_cross_dim),
            hidden_sizes=[self.node_feature_dim])
        self.graph_aggregation_layer = MLP(feature_dim=self.node_feature_dim*2, hidden_sizes=[1], softmax=True)

    def forward(self, graph_1_node, graph_2_node):
        # node embedding
        node_embedding_1 = self.Node_embedding(graph_1_node)
        node_embedding_2 = self.Node_embedding(graph_2_node)
        # print(f"embedding = {node_embedding_1[1, 1]}")
        # edge embedding
        graph_1_edg = create_edg(graph_1_node)
        graph_2_edg = create_edg(graph_2_node)
        edg_embedding_1 = self.edge_embedding(graph_1_edg).view(self.node_num, self.node_num, -1)
        edg_embedding_2 = self.edge_embedding(graph_2_edg).view(self.node_num, self.node_num, -1)
        # print(f"edg embedding = {edg_embedding_1[1, 1, 1]}")
        # Propagation_layer
        graph_1_intra = graph_Propagation_layer(graph_1_node, edg_embedding_1, self.graph_propagation_layer)
        graph_2_intra = graph_Propagation_layer(graph_2_node, edg_embedding_2, self.graph_propagation_layer)
        # print(f"intra = {graph_1_intra[1, 1]}")
        # Interaction layer
        graph_1_cross, graph_2_cross = graph_interaction_layer(node_embedding_1, node_embedding_2)
        # print(f"cross = {graph_1_cross[1, 1]}")
        # Update layer
        graph_1_update = self.graph_update_layer(torch.cat((F.normalize(node_embedding_1, p=2, dim=1),
                                                            F.normalize(graph_1_intra, p=2, dim=1),
                                                            F.normalize(graph_1_cross, p=2, dim=1)), dim=1))
        graph_2_update = self.graph_update_layer(torch.cat((F.normalize(node_embedding_2, p=2, dim=1),
                                                            F.normalize(graph_2_intra, p=2, dim=1),
                                                            F.normalize(graph_2_cross, p=2, dim=1)), dim=1))
        # print(f"update = {graph_1_update[1, 1]}")
        #  aggregation layer
        result_graph_1 = torch.sum(self.graph_aggregation_layer
                                   (torch.cat((graph_1_update, (torch.mean(graph_1_update, dim=0))
                                               .repeat(self.node_num, 1)), dim=1))*graph_1_update, dim=0).unsqueeze(0)
        result_graph_2 = torch.sum(self.graph_aggregation_layer
                                   (torch.cat((graph_2_update, (torch.mean(graph_2_update, dim=0))
                                               .repeat(self.node_num, 1)), dim=1)) * graph_2_update, dim=0).unsqueeze(0)
        # temp = self.graph_aggregation_layer(torch.cat((graph_2_update, (torch.mean(graph_2_update, dim=0))
        #                                        .repeat(self.node_num, 1)), dim=1))
        # print(f"shape = {temp}")

        # print(f"result = {result_graph_1[0, 1]}")
        # result_graph_1 = torch.softmax(result_graph_1, dim=1)
        # result_graph_2 = torch.softmax(result_graph_2, dim=1)
        similarity_score = (torch.cosine_similarity(result_graph_1, result_graph_2, dim=1) + 1) / 2
        # print(f"similarity = {similarity_score}")
        # print(similarity_score)
        return similarity_score


# model = GraphMatch(10, 128).cuda()
# node_1 = torch.normal(mean=0, std=1, size=(10, 128), dtype=torch.float, requires_grad=True).cuda()
# node_2 = torch.normal(mean=0, std=1, size=(10, 128), dtype=torch.float, requires_grad=True).cuda()
# edg_1 = torch.ones(size=(10, 128))
# edg_2 = torch.ones(size=(10, 128))
#
# optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-6, momentum=0.99, weight_decay=1e-5)
# for i in range(250):
#     result = model(node_1, node_2)
#     # print(f"result = {result}")
#     loss = 1 - result
#     # print(f"loss = {loss}")
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# test_1 = torch.normal(mean=0, std=1, size=(10, 128), dtype=torch.float).cuda()
# test_2 = torch.normal(mean=0, std=1, size=(10, 128), dtype=torch.float).cuda()
# print(model(test_1, test_2))


