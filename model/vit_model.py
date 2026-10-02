import torch
import torch.nn as nn
from torch.nn import LayerNorm
from collections import OrderedDict


class PatchEmbed(nn.Module):
    def __init__(self, img_size = 224, patch_size = 16, in_chans = 3, norm_layer = None):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)

        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.patch_nums = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = self.patch_size[0] * self.patch_size[1] * in_chans

        self.proj = nn.Conv2d(in_channels = in_chans, out_channels = self.embed_dim,
                              kernel_size = self.patch_size, stride = self.patch_size, bias = False)

        self.norm = LayerNorm(self.embed_dim) if norm_layer  else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)     #[batch, patch_num, embed_dim]
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim,num_heads = 8,qkv_bias = False,
                 qk_scale = None,attn_drop = 0.,proj_drop = 0.):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias = qkv_bias)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        batch, patch_nums, dim = x.shape
        qkv = self.qkv(x).reshape(batch, patch_nums, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]    #[b, heads, patch, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale   #[b, heads, patch, patch]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch, patch_nums, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class mlp(nn.Module):
    def __init__(self, in_dims, hidde_dim_ratio = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        hidden_dims = int(in_dims * hidde_dim_ratio) if hidde_dim_ratio is not None else in_dims
        self.fc1 = nn.Linear(in_features = in_dims, out_features = hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features = hidden_dims, out_features = in_dims)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class block(nn.Module):
    def __init__(self,in_chans,num_heads = 8,qkv_bias = False,
                 qk_scale = None,attn_drop = 0.,proj_drop = 0.,
                 hidde_dim_ratio = 4.0, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_chans)
        self.norm2 = nn.LayerNorm(in_chans)
        self.attn = Attention(in_chans, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.mlp = mlp(in_chans, hidde_dim_ratio, act_layer, drop)

    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x



class VisionTransformer(nn.Module):
    def __init__(self,num_classes, img_size = 224, patch_size = 16, in_chans = 3, norm_layer = None,
                 depth = 12, num_heads = 8, qkv_bias = False,
                 qk_scale = None, attn_drop = 0., proj_drop = 0.,
                 hidde_dim_ratio = 4.0, act_layer = nn.GELU, drop = 0.,
                 representation_size=None):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, norm_layer)
        self.patch_dim = self.patch_embed.embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,self.patch_embed.patch_nums + 1,self.patch_dim))

        self.blocks = nn.ModuleList([
            block(self.patch_dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop,hidde_dim_ratio,act_layer,drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.patch_dim)

        if representation_size is not None:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(self.patch_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.num_features = self.patch_dim
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)                                 #[b, patch_num, embed_dim]
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  #[b, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)            #[b, patch_num + 1, embed_dim]
        x = x + self.pos_embed

        for Block in self.blocks:
            x = Block(x)

        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        return x

    def forward(self,x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def vit_base_patch16_224_in21k(num_classes, has_logits=False):
    return VisionTransformer(
        num_classes=num_classes,
        img_size=224,
        patch_size=16,
        in_chans=3,
        norm_layer=None,
        depth=12,
        num_heads=12,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        hidde_dim_ratio=4.0,
        act_layer=nn.GELU,
        drop=0.,
        representation_size=768 if has_logits else None,
    )