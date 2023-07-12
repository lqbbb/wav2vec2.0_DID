import torch
from wenet.transformer.encoder_cat import ConformerEncoder
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

import torch.nn.functional as F

class Conformer(torch.nn.Module):
    def __init__(self, n_mels=83, num_blocks=6, output_size=256, input_layer="conv2d2",
            pos_enc_layer_type="rel_pos"):

        super(Conformer, self).__init__()
        print("input_layer: {}".format(input_layer))
        print("pos_enc_layer_type: {}".format(pos_enc_layer_type))
        self.conformer = ConformerEncoder(input_size=n_mels, num_blocks=num_blocks, 
                output_size=output_size, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)



    
    def forward(self, feat):
        feat = feat.transpose(1, 2)
        # feat = feat.squeeze(1).permute(0, 2, 1)
        feat = feat.squeeze(1)
        lens = torch.ones(feat.shape[0]).to(feat.device)
        lens = torch.round(lens*feat.shape[1]).int()
        x, masks = self.conformer(feat, lens)
        # print("x=",x.shape)
        x = x.permute(0, 2, 1)
        # x = self.pooling(x)
        # x = self.bn(x)
        # x = x.permute(0, 2, 1)
        # x = self.fc(x)
        # x = x.squeeze(1)



        return x

def conformer_cat(num_classes=17,n_mels=83, num_blocks=6, output_size=256,
        embedding_dim=192, input_layer="conv2d", pos_enc_layer_type="rel_pos"):
    model = Conformer(num_classes=num_classes, n_mels=n_mels, num_blocks=num_blocks, output_size=output_size,
            embedding_dim=embedding_dim, input_layer=input_layer, pos_enc_layer_type=pos_enc_layer_type)
    return model

 
if __name__ == '__main__':
    # Input size: batch_size  * feat_dim * seq_len
    x = torch.rand(4,300,80)    #torch.zeros()返回一个由标量值0填充的张量
    model = conformer_cat()
    out = model(x)

    # print(model)
    print(out.shape)    # should be [2, 192]
