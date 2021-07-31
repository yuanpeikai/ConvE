import torch
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_
import lib

class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, lib.embedding_dim, padding_idx=0)  # 200
        self.emb_rel = torch.nn.Embedding(num_relations, lib.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(lib.input_drop)  # 0.2
        self.hidden_drop = torch.nn.Dropout(lib.hidden_drop)  # 0.3
        self.feature_map_drop = torch.nn.Dropout2d(lib.feat_drop)  # 0.2

        self.emb_dim1 = lib.embedding_shape1  # 20
        self.emb_dim2 = lib.embedding_dim // self.emb_dim1  # 200//20=10

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=lib.use_bias)  # true
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(lib.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(lib.hidden_size, lib.embedding_dim)
        self.init()
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded = self.emb_e(e1).view(-1, 1, self.emb_dim1, self.emb_dim2)  # batch_size,1,20,10
        rel_embedded = self.emb_rel(rel).view(-1, 1, self.emb_dim1, self.emb_dim2)  # batch_size,1,20,10

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)  ##batch_size,1,40,10

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)#batch_size,32,38,8  #他的卷积核是[32, 1, 3, 3] bias:[32]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)  # batch_size,38*8*32
        x = self.fc(x)  # batch_size,200
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1, 0))  # batch_size,num_entities
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred
