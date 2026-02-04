import torch
from torch import nn, einsum
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class Conv4ST(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, in_length=10, out_length=10, hid_s=8,  n_s=4, n_t=8, dropout=0.3, scale_t=8):
        super().__init__()
        self.spatial_enc =  SpatialEncoder(out_channels, hid_s, n_s)
        self.hid = hid_s * 2**n_s
        self.temporal_enc = TemporalEncoder(in_length, out_length, self.hid, n_t, dropout=dropout, scale_t=scale_t)
        self.spatial_dec = SpatialDecoder(self.hid, out_channels, n_s)

        self.total_length = in_length + out_length
       
        self.temporal_dec = nn.Sequential(nn.GroupNorm(1,self.total_length),
                                          nn.Conv2d(self.total_length, self.total_length*scale_t, kernel_size=3, stride=1, padding=1),
                                          nn.GroupNorm(1,self.total_length*scale_t),nn.PReLU(),
                                          nn.Conv2d(self.total_length*scale_t, out_length, kernel_size=3, stride=1, padding=1),
                                          )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, T_in, C, H, W]
        B, T_in, C, H, W = x.shape

        raw =  x.clone()
        raw = rearrange(raw, 'b t c h w -> (b c) t h w')

        x = rearrange(x, 'b t c h w -> (b t) c h w', t=T_in)

        x = self.spatial_enc(x) # [B*T_in, hid=hid_s*2^n_s, h=H//2^n_s, w=W//2^n_s]

        x = rearrange(x, '(b t) c h w -> (b c) t h w', t=T_in)  # [B*hid, T_in, h, w]

        x = self.temporal_enc(x) # [B*hid, T_out , h, w]

        x = rearrange(x, '(b c) t h w -> (b t) c h w', b=B)

        x = self.spatial_dec(x) # [B*T_out, out_channels, H, W]

        # x = rearrange(x, '(b t) c h w -> (b c) t h w', b=B)
        x = rearrange(x, '(b t) c h w -> (b c) t h w', b=B)

        # x = self.temporal_dec(x+raw)
        x = torch.cat([x,raw], dim=1)
        x = self.temporal_dec(x)
        x = rearrange(x, '(b c) t h w -> b t c h w', b=B, t=self.total_length-T_in)

        return x

def sampling_generator(N, reverse=False):
    samplings = [False, True] * N
    if reverse: return list(reversed(samplings))
    else: return samplings

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, hid_s, n_s):
        samplings = sampling_generator(n_s)
        super().__init__()
        f = lambda i: 2**(i//2) if i//2 != 0 else 1
        self.enc = nn.Sequential(
            ConvBlock(in_channels, hid_s, 3, downsampling=samplings[0], ),
            *[ConvBlock(hid_s * f(i+1) , hid_s * f(i+2), 3, downsampling=s,)
              for i, s in enumerate(samplings[1:])]
        )

    def forward(self, x):
        for i in range(len(self.enc)):
            x = self.enc[i](x)
        return x


class SpatialDecoder(nn.Module):
    def __init__(self, hid, out_channels,n_s):
        super().__init__()
        self.upsampling = nn.Sequential(nn.Conv2d(hid, out_channels*4**n_s, 1),nn.PixelShuffle(2**n_s))

    def forward(self, x):
        x = self.upsampling(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, in_length, out_length, hid,  n_t, kernel_size=3, dropout=0., scale_t=8):
        super().__init__()
        self.in_length = in_length
        self.temporal_encoder_modules = nn.ModuleList(
            [TemporalModule(in_length, out_length, hid,  dropout=dropout, is_first= i==0,
                                   kernel_size=kernel_size, dilation=1, scale_t=scale_t) for i in range(n_t)])
    def forward(self, x):
        latent = x.clone()
        for module in self.temporal_encoder_modules:
            x = module(x, latent)
        return x

class TemporalModule(nn.Module):
    def __init__(self, in_length, out_length, hid,  kernel_size=3, dilation=1, dropout=0.,is_first=False, scale_t=8):
        super().__init__()
        self.total_length = in_length + out_length
        self.is_first = is_first
        if is_first:
            self.temporal_embedding = Conv(in_length, out_length)
        self.temporal_proj = nn.Sequential(
            Conv(self.total_length, scale_t*self.total_length, kernel_size=kernel_size,dilation=dilation,dropout=dropout),
            nn.Conv2d(scale_t*self.total_length, out_length, kernel_size=3, stride=1, padding=1))
        self.mixing = nn.Sequential(
            Conv(hid, scale_t*hid, dropout=dropout),
            nn.Conv2d(scale_t*hid, hid, kernel_size=3, stride=1, padding=1))
        self.hid = hid
        self.out_length = out_length

    def forward(self, x, latent):
        if self.is_first:
            x = self.temporal_embedding(x)  # [B*c, T_out, h, w]
        x = torch.cat([latent, x], dim=1)  # [B*c, T_in+T_out, h, w]
        x = self.temporal_proj(x) # [B*c, T_out, h, w]
        x = rearrange(x, '(b c) t h w -> (b t) c h w',c=self.hid)
        x = self.mixing(x)
        x = rearrange(x, '(b t) c h w -> (b c) t h w', t=self.out_length)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  dilation=1,  dropout=0., norm=True, group=1):
        super().__init__()
        self.proj = in_channels != out_channels
        if self.proj:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.norm = norm
        if norm:
            self.norm1 = nn.GroupNorm(group,out_channels)
            self.norm2 = nn.GroupNorm(group,out_channels)
        # self.act = nn.GELU()
        self.act1 = nn.PReLU()
        self.act2 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        if self.proj:
            skip = self.proj(x)
        else:
            skip = x.clone()
        x = self.conv(x)
        if self.norm:
            x = self.norm1(x)
        x = self.act1(x)
        if self.dropout.p > 0:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.norm:
            x = self.norm2(x)
        x = self.act2(x)
        x = x + skip
        return x




class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True):
        super().__init__()

        stride = 2 if downsampling else 1

        padding = (kernel_size - stride + 1) // 2

        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=kernel_size,
                          stride=1, padding=padding),
                nn.PixelShuffle(2)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding)

        self.act = nn.PReLU()
        self.norm = nn.BatchNorm2d(out_channels)
        self.act_norm = act_norm

    def forward(self, x):
        x = self.conv(x)
        if self.act_norm:
            x = self.act(self.norm(x))
        return x

if __name__ == '__main__':
    x = torch.randn(3, 10, 1, 160, 160).to('cuda')
    model = Conv4ST(1, 1,10,10,hid_s=16, n_t=16,n_s=3).to('cuda')
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"total parameters: {total_params}\n"
          f"trainable paras: {trainable_params}")
    y = model(x)
    print(y.shape)

