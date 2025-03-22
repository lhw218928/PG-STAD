import torch.nn as nn
import torch
from modules import TimeFeatureLayer,StructureFeatureLayer, ForecastModule, ReconstructionModule
class PT_STAD(nn.Module):
    def __init__(
            self,
            features_num,
            window_size,
            window_num,
            target_dims,
            structure_feature_embed_dim = None,
            use_gatv2 = True,
            gru_layers = 1,
            time_feature_embed_dim = None,
            forecast_hidden_dim = 32,
            forecast_n_layers =1,
            recon_hid_dim=150,
            recon_n_layers = 1,
            dropout=0.1,
            alpha=0.2
    ):
        super(PT_STAD, self).__init__()

        self.structureAttentionLayer = StructureFeatureLayer(features_num, window_size, dropout, alpha, structure_feature_embed_dim, use_gatv2)
        self.timeAttentionLayer = TimeFeatureLayer(window_size, time_feature_embed_dim, gru_layers, dropout)
        self.concat_feature_dim = 2 * window_size
        self.forecastModule = ForecastModule(window_size, self.concat_feature_dim, forecast_hidden_dim, target_dims, forecast_n_layers, dropout)  #将nfeature换成window_size,多步window_size
        self.reconstructionModule = ReconstructionModule(window_size, features_num * self.concat_feature_dim, recon_hid_dim, target_dims, recon_n_layers, dropout)

    def forward(self,x):
        # x shape (b, k*w, n): b - batch size, w - window size, k - window num, n - number of features

        structedFeatures = self.structureAttentionLayer(x)   #输出结果[b,n,w]
        timeFeatures = self.timeAttentionLayer(x)    #返回结果[b,n,w]

        timeFeatures_hidden = timeFeatures[1]  #形状是[b,num_layers*n,w]

        Features = torch.cat([structedFeatures, timeFeatures_hidden],dim=2)    #残差连接可以考虑[b,n,2*W]

        # Pred_Features = Features.view(-1,Features.size(-1))

        predictions = self.forecastModule(Features)  #[b,out_dim,w]
        predictions = predictions.permute(0,2,1) #[b,w,out_dim]


        recons = self.reconstructionModule(Features)


        # 在经过gru前，应该变换成bn,k,w的格式
        # x的输入格式是[batch_size,sequence_length,input_size],
        # 输出格式是[num_layers * num_directions, batch_size, hidden_size]
        # 经过gru后，输出形式就是bn，hidden_size的格式，再view一下


        return predictions,recons