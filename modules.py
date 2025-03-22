import numpy as np
import torch.nn as nn
import torch
class StructureFeatureLayer(nn.Module):

    """Structure Feature Extraction Layer
        :param features_num: Number of input features/nodes
        :param window_size: length of the input sequence
        :param dropout: percentage of nodes to dropout
        :param alpha: negative slope used in the leaky relu activation function
        :param structure_feature_embed_dim: embedding dimension (output dimension of linear transformation)
        :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
        :param use_bias: whether to include a bias term in the attention layer
        """

    def __init__(self, features_num , window_size, dropout, alpha, structure_feature_embed_dim = None, use_gatv2 = True, use_bias = True):
        super(StructureFeatureLayer, self).__init__()
        self.features_num = features_num
        self.window_size = window_size
        self.dropout = dropout
        self.structure_feature_embed_dim = structure_feature_embed_dim if structure_feature_embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.nodes_num = features_num
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.structure_feature_embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.structure_feature_embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.structure_feature_embed_dim

        self.lin = nn.Linear(lin_input_dim, self.structure_feature_embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(features_num, features_num))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, k*w, n): b - batch size, w - window size, k - window num, n - number of features
        # We utilize the data features from the last sliding window to construct the adjacency matrix, integrating structural information.

        x = x[:,-self.window_size:,:]

        # Causal_Matrix = batch_granger_causality(data=x, max_lag=self.window_size)

        x = x.permute(0, 2, 1)   #shape变为[b,n,w]

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)     相当于算出了注意力系数
        # Original GAT attention
        else:
            Wx = self.lin(x)  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, n, n, 1)
        if self.use_bias:
            e += self.bias

        # e = np.multiply(e,Causal_Matrix)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)  # [b,n,n]

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))  # [b,n,n] 乘 [b,n,w]

        return h #h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        N = self.nodes_num
        blocks_repeating = v.repeat_interleave(N, dim=1)  # Left-side of the matrix
        # blocks_repeating (b,K*K,window_size)
        blocks_alternating = v.repeat(1, N, 1)  # Right-side of the matrix
        # blocks_alternating (b,K*K,window_size)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), N, N, 2 * self.window_size)
        else:
            return combined.view(v.size(0), N, N, 2 * self.structure_feature_embed_dim)

class TimeFeatureLayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """
    # in_dim 应该等于window_size或者window_size做一维卷积的结果(如果window_size过大的话)
    def __init__(self, in_dim, hid_dim, n_layers, dropout): #in_dim输入特征数量
        super(TimeFeatureLayer, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim if hid_dim is not None else in_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(self.in_dim, self.hid_dim, num_layers=self.n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):    #x[b,k*w,n]
        batch_size = x.size(0)
        x = x.permute(0,2,1).contiguous()    #x[b,n,k*w]
        x = x.view(-1,x.size(-1)) #x[b*n,k*w]
        x = x.view(x.size()[0],-1,self.in_dim)  #x[b*n,k,w]
        out, h = self.gru(x)     #out[b*n,k,hid_dim]  h[n_layers,b*n,hid_dim]
        out = out[:,-1,:]
        # out, h = out[-1, :, :], h[-1, :, :]  # Extracting from last layer  一般hidden_dim=window_size
        out ,h = out.view(batch_size,-1,self.hid_dim),h.view(batch_size,-1,self.hid_dim)   #return_size:[b,n,hid_dim]
        return out, h

class ForecastModule(nn.Module):
    """Forecasting model (fully-connected network)
        :param in_dim: number of input features
        :param hid_dim: hidden size of the FC network
        :param out_dim: number of output features
        :param n_layers: number of FC layers
        :param dropout: dropout rate
        """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        self.out_dim = out_dim
        super(ForecastModule, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, window_size))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):   #[b,n,2w]    out_dim其实就是要预测的变量个数
        x = x[:,:self.out_dim,:]
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)

class ReconstructionModule(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param out_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModule, self).__init__()

        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x [b,n,2*w]   需要去重构这个时间窗口内的值，输出维度是[b,out_dim,w]

        x = x.view(x.size(0),-1)
        x = x.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size,-1)
        decoder_out = self.decoder(x)

        # h_end = x
        # h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        #
        # decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out

class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        decoder_out, _ = self.rnn(x)
        return decoder_out