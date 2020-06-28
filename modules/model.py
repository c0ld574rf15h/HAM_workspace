import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import read_config


class Attention(nn.Module):

    def __init__(self, of):
        super().__init__()
        self.conf = read_config('model')

        self.linear_1 = nn.Linear(
            in_features = eval(self.conf['{}GRUHiddenDim'.format(of)]) * 2,
            out_features = eval(self.conf['{}LinearHiddenDim'.format(of)])
        )

        self.linear_2 = nn.Linear(
            in_features = eval(self.conf['{}LinearHiddenDim'.format(of)]),
            out_features = 1
        )

    def forward(self, attention_in):
        hidden = torch.tanh(self.linear_1(attention_in))
        attention_out = self.linear_2(hidden)

        return F.softmax(attention_out, dim=1)


class HierarchicalAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conf = read_config('model')

        # Byte Level
        self.byte_embedding = nn.Embedding(256, eval(self.conf['ByteEmbeddingDim']))

        self.byte_biGRU = nn.GRU(
            input_size=eval(self.conf['ByteEmbeddingDim']),
            hidden_size=eval(self.conf['ByteGRUHiddenDim']),
            bidirectional=True
        )

        self.byte_attention = Attention(of='Byte')

        # Packet Level
        self.packet_biGRU = nn.GRU(
            input_size=eval(self.conf['PacketEmbeddingDim']),
            hidden_size=eval(self.conf['PacketGRUHiddenDim']),
            bidirectional=True
        )

        self.packet_attention = Attention(of='Packet')

        # Final Classification
        self.final_classification = nn.Linear(
            in_features=eval(self.conf['PacketGRUHiddenDim']) * 2,
            out_features=eval(self.conf['NumFlowClass'])
        )

    def forward(self, flow):
        num_packets, batch_sz = flow.shape[0], flow.shape[1]
        packet_embeddings = torch.zeros(
            num_packets,
            batch_sz,
            eval(self.conf['PacketEmbeddingDim'])
        ).to('cuda')

        # Byte Level
        byte_embeddings = self.byte_embedding(flow)

        for idx, byte_embedding in enumerate(byte_embeddings):
            init_h = torch.zeros((2, batch_sz, eval(self.conf['ByteGRUHiddenDim']))).to('cuda')
            output, _ = self.byte_biGRU(byte_embedding.transpose(0, 1), init_h)

            attention_scores = self.byte_attention(output)
            packet_embeddings[idx, :, :] = (output * attention_scores).sum(dim=0)

        # Packet Level
        init_h = torch.zeros((2, batch_sz, eval(self.conf['PacketGRUHiddenDim']))).to('cuda')
        output, _ = self.packet_biGRU(packet_embeddings, init_h)

        attention_scores = self.packet_attention(output)
        flow_embedding = (output * attention_scores).sum(dim=1)

        # Final Classification
        preds = self.final_classification(flow_embedding)

        return preds
