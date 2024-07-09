import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class WSIEncoder(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        B, _, _ = x.shape
        for b in range(B):
            x[b]= x[b] + self.attn(self.norm(x[b]))
        return x

class PatchEncoder(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=1, dim_feedforward=2048):
        super(PatchEncoder, self).__init__()
        self.input_dim = dim
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        B, _, _= x.shape
        for b in range(B):
            xb = x[b]  # shape: [n, 512]
            xb = self.transformer_encoder(xb)
            x[b] = xb
        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

class GatedAttention(nn.Module):
    def __init__(self, dim):
        self.theta1 = torch.randn(dim, requires_grad=True)
        self.theta2 = torch.randn(dim, requires_grad=True)
        self.W = torch.randn(1, requires_grad=True)

    def calculate_attention_weights(self, f_nl):
        term1 = torch.tanh(torch.matmul(f_nl, self.theta1))
        term2 = torch.sigmoid(torch.matmul(f_nl, self.theta2))
        attention_score = torch.exp(self.W * (term1 * term2))
        return attention_score

    def aggregate_features(self, x):
        B, N, C = x.shape

        atten = torch.zeros(B,N)

        for b in range(B):
            for n in range(N):
                xp = x[b,n]
                atten[b, n] = self.calculate_attention_weights(xp)
        atten /= torch.sum(atten)
        for b in range(B):
            for n in range(N):
                xp = x[b, n]
                x[b,n] = atten[b,n] * xp

        return x,atten

class PCS_Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features, norm_scale, a_margin):
        super(PCS_Classifier, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.norm_scale = norm_scale
        self.a_margin = a_margin
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)


    def forward(self, x,t,atten,train):
        B, N, C = x.shape
        weight_norm = F.normalize(self.weight, dim=0)
        scores = torch.mean(atten,dim=1)/torch.max(atten,dim=1)
        outputs = torch.zeros(B,self.out_features)
        outputs_modified = torch.zeros(B,self.out_features)
        for b in range(B):
            x_norm = F.normalize(x[b])
            cos = torch.mm(x_norm, weight_norm)
            cos = cos.clamp(-1, 1)
            cost = cos[t[b]]
            margin = self.a_margin*torch.pow((1.0-scores[b]),2)

            if train and cost > 0:
                sint = torch.sqrt(1.0 - torch.pow(cost, 2))
                a = cost * torch.cos(margin) - sint * torch.sin(margin)

                cos_modified = cos.clone()
                cos_modified[t] = a
                outputs_modified[b]=cos_modified*self.norm_scale
            else:
                outputs_modified[b]=cos*self.norm_scale

            outputs[b]=cos*self.norm_scale

        return outputs,outputs_modified


class PSJAMIL(nn.Module):
    def __init__(self, n_classes, norm_scale, a_margin):
        super(PSJAMIL, self).__init__()

        self.n_classes = n_classes

        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.trans1 = PatchEncoder(dim=512)  # PatchEncoder

        self.atten = GatedAttention(dim=512)

        self.trans2 = WSIEncoder(dim=512)  # WSIEncoder
        self.pos_layer = PPEG(dim=512)
        self.trans3 = WSIEncoder(dim=512)  # WSIDecoder

        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, 256)
        self._fc3 = PCS_Classifier(256,self.n_classes,norm_scale, a_margin)


    def forward(self, **kwargs):
        h = kwargs['data'].float() #[B, n, 1024]

        # ---->Patch modeling
        h = self._fc1(h) #[B, n, 512]
        fea = self.trans1(h)

        # ---->Attention modeling
        h, atten = self.atten(fea)

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->WSI modeling
        h = self.trans2(h)  # [B, N, 512]
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        h = self.trans3(h)  # [B, N, 512]

        # ---->predict
        h = self.norm(h) # [B, N, 512]
        h = torch.mean(h, dim=1, keepdim=True) # [B, 1, 512]
        h = self._fc2(h) # [B, 1, 256]
        outputs,logits = self._fc3(h,int(kwargs['label']),atten,kwargs['train']) #[B, n_classes]

        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits_ori': outputs,'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,
                        'atten_score': atten, 'fea': fea}

        return results_dict

if __name__ == "__main__":
    data = torch.randn((16, 6000, 1024)).cuda()
    model = PSJAMIL(n_classes=3).cuda()
    results_dict, h = model(data = data)


