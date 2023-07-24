import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm.models.vision_transformer
from timm.models.layers import DropPath, trunc_normal_, Mlp

from lib.utils.misc import is_main_process
from lib.models.mixformer.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.mixformer_vit.pos_util import get_2d_sincos_pos_embed

from itertools import repeat
import collections.abc

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # BCHW -> BNC
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, num_t, num_s):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_t, q_s = torch.split(q, [num_t, num_s], dim=2)
        k_t, k_s = torch.split(k, [num_t, num_s], dim=2)
        v_t, v_s = torch.split(v, [num_t, num_s], dim=2)

        # asymmetric mixed attention
        attn = (q_t @ k_t.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_t = (attn @ v_t).transpose(1, 2).reshape(B, num_t, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, num_s, C)

        x = torch.cat([x_t, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pool = torch.nn.AvgPool2d(2, stride=2)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, num_t, num_s, tokens_pooling=False, unpooling_shortcut=None, unpooling_rate=2):
        # Tokens unpooling & add shortcut
        if unpooling_shortcut is not None:
            x_t, x_s = torch.split(x, [num_t, num_s], dim=1)
            w_s = h_s = int(num_s ** 0.5)
            assert w_s * h_s == num_s
            x_s = x_s.reshape(x.size(0), h_s, w_s, -1)
            x_s_unpool = F.interpolate(x_s.permute(0,3,1,2), scale_factor=unpooling_rate, mode='bilinear')
            x_s_unpool = x_s_unpool.contiguous().permute(0,2,3,1)
            unpooling_shortcut = unpooling_shortcut.reshape(x_s_unpool.shape)
            x_s = (x_s_unpool + unpooling_shortcut).view(x.size(0), -1, x.size(-1))
            num_s = num_s * (unpooling_rate ** 2)
            x = torch.cat([x_t, x_s], dim=1)

        # Tokens pooling
        if tokens_pooling:
            x_t, x_s = torch.split(x, [num_t, num_s], dim=1)
            w_s = h_s = int(num_s ** 0.5)
            assert w_s * h_s == num_s
            x_s = x_s.reshape(x.size(0), h_s, w_s, -1)
            x_s_pool = self.pool(x_s).reshape(x.size(0), -1, x_s.size(-1))
            num_s = x_s_pool.size(1)
            x = torch.cat([x_t, x_s_pool], dim=1)

        x = x + self.drop_path1(self.attn(self.norm1(x), num_t, num_s))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size_s=256, img_size_t=128, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super(VisionTransformer, self).__init__(img_size=224, patch_size=patch_size, in_chans=in_chans,
                                                num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                                                drop_path_rate=drop_path_rate, weight_init=weight_init,
                                                norm_layer=norm_layer, act_layer=act_layer)

        self.patch_embed = embed_layer(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])

        self.num_patches_s = (img_size_s // patch_size) ** 2
        self.num_patches_t = (img_size_t // patch_size) ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim), requires_grad=False)

        self.initialize_pos_weights()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def initialize_pos_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                            cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_s, pooling_layer=3, unpooling_layer=9, pooling_size=2):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 256, 256)
        :return:
        """
        x_t = self.patch_embed(x_t)  # BCHW-->BNC
        x_s = self.patch_embed(x_s)
        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = int(math.sqrt(x_s.size(1)+0.1))
        H_t = W_t = int(math.sqrt(x_t.size(1)+0.1))

        # pos_s = self.pos_embed[:, :-1, :]   # delete the initial cls_toke in ViT
        # pos_s_2d = pos_s.transpose(1,2).reshape(pos_s.size(0), C, H_s, W_s)
        # pos_t_2d = F.interpolate(pos_s_2d, size=(H_t, W_t), mode='bilinear')
        # pos_t = pos_t_2d.flatten(2).transpose(1,2) # BNC

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x = torch.cat([x_t, x_s], dim=1)
        # x = x + self.pos_embed
        x = self.pos_drop(x)

        len_s = H_s*W_s

        layer_id = 0
        unpooling_shortcut = None
        num_t = H_t * W_t
        num_s = H_s * W_s
        for blk in self.blocks:
            if layer_id == pooling_layer:
                # tokens pooling
                x_t, x_s = torch.split(x, [num_t, num_s], dim=1)
                w_s = h_s = int(num_s ** 0.5)
                x_s = x_s.reshape(x.size(0), h_s, w_s, -1)
                unpooling_shortcut = x_s
                x = blk(x, num_t, num_s, tokens_pooling=True)
                num_s = num_s // (pooling_size ** 2)
            elif layer_id == unpooling_layer:
                # tokens unpooling
                x = blk(x, num_t, num_s, unpooling_shortcut=unpooling_shortcut)
                num_s = num_s * (pooling_size ** 2)
            else:
                x = blk(x, num_t, num_s)
            layer_id += 1

        # for blk in self.blocks:
        #     x = blk(x, H_t * W_t, H_s * W_s)

        x_t, x_s = torch.split(x, [H_t*W_t, len_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_s_2d


def get_mixformer_sparse_vit(config):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    if config.MODEL.VIT_TYPE == 'large_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    elif config.MODEL.VIT_TYPE == 'base_patch16':
        vit = VisionTransformer(
            img_size_s=img_size_s, img_size_t=img_size_t,
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1)
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'large_patch16' or 'base_patch16'")

    if config.MODEL.BACKBONE.PRETRAINED:
        # try:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        ckpt = torch.load(ckpt_path, map_location='cpu')['model'] #['net']  #image_mae:['model']
        # interpolate pos embedding of search and template
        # new_dict = {}
        # H_s = W_s = img_size_s // 16
        # H_t = W_t = img_size_t // 16
        # for k, v in ckpt.items():
            # if 'pos_embed' in k:
            #     new_dict[k] = v
            #     pos = ckpt[k]
            #     C = pos.size(2)
            #     pos_2d = pos[:, :-1, :].transpose(1,2).reshape(1, C, 14, 14)
            #     pos_s_2d = F.interpolate(pos_2d, size=(H_s, W_s), mode='bilinear')
            #     pos_s = pos_s_2d.flatten(2).transpose(1,2) # BNC
            #     pos_t_2d = F.interpolate(pos_2d, size=(H_t, W_t), mode='bilinear')
            #     pos_t = pos_t_2d.flatten(2).transpose(1,2) # BNC
            #     pos_s_key = k.replace('pos_embed', 'pos_embed_s')
            #     pos_t_key = k.replace('pos_embed', 'pos_embed_t')
            #     new_dict[pos_s_key] = pos_s
            #     new_dict[pos_t_key] = pos_t
            #     print("Interpolating current pos_embed_s and pos_embed_t using {} done.".format(k))
            # if 'pos_embed_t' in k:
            #     new_dict[k] = v
            #     pos_t = ckpt[k]
            #     C = pos_t.size(2)
            #     pos_t_2d = pos_t[:, :-1, :].transpose(1,2).reshape(1, C, H_s, W_s)   #TODO：modify the size
            #     pos_t_2d = F.interpolate(pos_t_2d, size=(H_t, W_t), mode='bilinear')
            #     pos_t = pos_t_2d.flatten(2).transpose(1,2) # BN
            #     new_dict[k] = pos_t
            #     print("Interpolating current pos_embed_t using {} done.".format(k))
            # else:
            #     new_dict[k] = v
        new_dict = {}
        for k, v in ckpt.items():
            if 'pos_embed' not in k and 'mask_token' not in k:
                new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")
        # except:
        #     print("Warning: Pretrained ViT weights are not loaded")

    return vit


class MixFormer(nn.Module):
    """ This is the base class for Transformer Tracking, whcih jointly perform feature extraction and interaction. """
    def __init__(self, backbone, box_head, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type

    def forward(self, template, online_template, search, run_score_head=False, gt_bboxes=None):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        # if online_template.dim() == 5:
        #     template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone(template, search)
        # Forward the corner head
        return self.forward_box_head(search)

    def forward_test(self, search, run_box_head=True, run_cls_head=False):
        # search: (b, c, h, w) h=20
        if search.dim() == 5:
            search = search.squeeze(0)
        search = self.backbone.forward_test(search)
        # search (b, 384, 20, 20)
        # Forward the corner head
        return self.forward_box_head(search)

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if self.head_type == "CORNER":
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        else:
            raise KeyError

def build_mixformer_sparse_vit(cfg):
    backbone = get_mixformer_sparse_vit(cfg)  # backbone without positional encoding and attention mask
    box_head = build_box_head(cfg)  # a simple corner head
    model = MixFormer(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model
