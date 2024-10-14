import math
import numpy as np
from timm.models.registry import register_model
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

"""
简单多层感知器  工作流程：
输入层：数据输入，特征维度为 in_features。
第一层全连接（fc1）：将输入特征映射到隐藏特征空间，维度从 in_features 变为 hidden_features。
激活函数（act）：通过激活函数（默认为 GELU）对隐藏特征进行非线性变换。
第二层全连接（fc2）：将隐藏特征映射到输出特征空间，维度从 hidden_features 变为 out_features。
Dropout：在输出前应用 Dropout 操作，减少过拟合。
输出：返回经过 MLP 处理的输出。
"""
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=True)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)  # 对输入特征进行归一化处理的层，使用 Layer Normalization
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # DropPath 是一种正则化技术，旨在通过在训练过程中随机丢弃整个路径来防止深度学习模型的过拟合。在标准的 dropout 中，我们通常丢弃一些神经元的输出，而 DropPath 则是在特定层的整个输出被随机选择丢弃，这意味着在前向传播中某些层的输出会被完全忽略
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 第二个归一化层，处理从 MLP 输出的特征
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        # 先归一化，再注意力处理得到更新后的特征，再随机丢弃路径，然后和X相加，形成残差连接
        # 再次归一化，然后用MLP，进一步处理特征，然后形成第二次残差链接
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

"""
PositionalEncoding 类是位置编码（Positional Encoding）的实现，主要用于将位置信息引入到模型中
尤其是在像 Transformer 这样没有显式的顺序信息的架构中。
位置编码用于补充自注意力机制（Self-Attention）的顺序信息，让模型能够区分序列中的不同位置
"""
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # 初始化一个形状为 (max_len, d_model) 的全零张量，用于存储每个位置的编码信息
        position = torch.arange(0, max_len).unsqueeze(1)  # 创建一个从 0 到 max_len 的序列，表示位置索引。维度为 [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) # math.log(math.exp(1)) = 1 计算用于正弦和余弦函数的分母部分。这个分母是一个随特征维度变化的指数衰减值，其目的是保证不同维度上的频率不同，从而编码不同的位置信息
        # 生成正弦和余弦编码
        # 正弦和余弦函数生成的编码值使得不同位置的编码是不同的，并且能够保留一定的相对位置信息。这种结构设计的目的是让模型能够通过相对位置捕捉顺序信息
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列使用正弦编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列使用余弦编码
        pe = pe.unsqueeze(0) # 为位置编码增加一个批次维度，[1, max_len, d_model]
        self.register_buffer('pe', pe)  # register_buffer: 将 pe 位置编码注册为缓冲区，不会作为模型参数参与训练，但会随着模型保存和加载
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 将输入 query，key，value 通过对应的线性层 wq, wk, wv，将其转换为新的维度（d_model），并 reshape 为 num_heads 个子向量，维度为 d_k
        # 通过 view 和 transpose 函数，将 query, key, value 的形状转换为 (nbatches, num_heads, seq_len, d_k)，其中 seq_len 是序列长度
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        # print(key.shape)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        # 进行多头注意力计算，返回新的值 x 和注意力权重 p_attn
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)  # 通过计算查询向量和键向量的点积，得到注意力分数
        #print(mask.shape)
        if mask is not None:
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)  # 使用 softmax 函数将分数转换为注意力权重，这些权重表示每个位置的重要性
        return torch.matmul(p_attn, value), p_attn

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)  # 多头自注意力机制，它接收输入序列 x，并计算每个元素对序列中其他元素的注意力分数
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout = 0.1)  # 跨注意力机制，它计算输入序列 x 与记忆 memory（通常是编码器输出）的注意力分数
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout = 0.1)  # 两层的前馈神经网络，类似于一个扩展和压缩的过程；第一个全连接层将维度从 d_model 扩展到 dff，然后通过激活函数，再通过另一个全连接层将维度压缩回 d_model

        # 层归一化的作用是帮助稳定训练，避免梯度消失和梯度爆炸问题
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "文章图5右侧的三层连接方式：多头自注意力+多头跨注意力+前馈神经网络"
        attn_output = self.self_mha(x, x, x, look_ahead_mask)  # look_ahead_mask 作为掩码（mask）用来防止模型查看未来的 tokens，常用于解码阶段防止模型作弊
        x = self.layernorm1(x + attn_output)  # 在每个子层后，进行残差连接（即输入 x 加上经过子层的输出，逐元素相加），然后进行层归一化

        """
        注解中的 q, k, v 分别代表 query（查询向量）、key（键向量） 和 value（值向量），它们是多头自注意力机制（Multi-Headed Attention）中的核心概念。这些向量之间的相互作用决定了注意力的计算方式。具体解释如下：
        Query (q): 查询向量用于“询问”其他位置的特征，查询当前位置与其他位置的相关性。
        Key (k): 键向量帮助查询向量找到与其相关的其他位置。键向量和查询向量的相似度决定了注意力的权重。
        Value (v): 值向量是实际要聚合的信息。在多头注意力中，计算出的注意力权重会应用于值向量，从而决定最终的注意力输出
        """
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x


class Decoder(nn.Module):
    def __init__(self, depth=4, embed_dim=128, num_heads=4, dff=128, drop_rate=0.1):
        """
        depth: 解码层的数量
        embed_dim: 嵌入的维度，每个输入特征的维度
        num_heads: 多头自注意力机制中的头数，通过并行计算多个注意力机制来提高模型的表达能力
        dff: 前馈网络的隐藏层维度，通常比嵌入维度大，便于处理非线性变换
        drop_rate: dropout 概率，用于防止过拟合
        """
        super(Decoder, self).__init__()
    
        self.d_model = embed_dim
        self.pos_encoding = PositionalEncoding(embed_dim, drop_rate, 50)  # 位置编码在序列模型中用于为模型提供输入序列中每个位置的信息，以帮助模型理解单词的顺序
        self.dec_layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, dff, drop_rate) 
                                            for _ in range(depth)])
        # 使用 nn.ModuleList 创建多个解码器层。这里每一层都是 DecoderLayer 的一个实例，并传入了 embed_dim、num_heads、dff 和 drop_rate

    """
    x: 输入特征，通常是从编码器获得的特征表示，形状为 [batch_size, target_sequence_length, embed_dim]。
    memory: 编码器输出的记忆（也称为上下文向量），它包含输入序列的信息。
    look_ahead_mask: 可选参数，用于防止模型在生成序列时看到未来的词（即未生成的部分），通常用于训练时的自回归模型。
    trg_padding_mask: 可选参数，用于遮挡目标序列中填充的部分，以确保模型在计算注意力时不考虑这些无用的填充部分
    """
    def forward(self, x, memory, look_ahead_mask=None, trg_padding_mask=None):
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)  # 每一层的 dec_layer 进行自注意力计算和前馈网络处理，输出新的特征表示 x。该过程的目的是在生成过程中结合目标序列的上下文信息
        return x  # 返回最终的 x，这通常是处理后的特征表示，可以用于后续的任务



from einops import rearrange, repeat     

def exists(val):
    return val is not None


class CNN_embedd(nn.Module):
    def __init__(self,
                 kernel_size=7, stride=2, padding=3,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=nn.ReLU,
                 max_pool=True,
                 conv_bias=False):
        super().__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        n_filter_list_pairs = zip(n_filter_list[:-1], n_filter_list[1:])

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(chan_in, chan_out,
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if not exists(activation) else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for chan_in, chan_out in n_filter_list_pairs
            ])

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return rearrange(self.conv_layers(x), 'b c h w -> b (h w) c')

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)






class ViTEncoder_imgcr(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
    
        self.CNN_embed = CNN_embedd(n_input_channels=3,
                                   n_output_channels=embed_dim)
        
        
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('img'):
            x = self.patch_embed(x)
            # x = self.CNN_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x


class ViTEncoder_vqa(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.linear_embed_vqa = nn.Linear(2048, self.embed_dim)

        # TODO: Add the cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('vqa'):
            x = self.linear_embed_vqa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x


class ViTEncoder_msa(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.linear_embed_msa = nn.Linear(35, self.embed_dim)

        # TODO: Add the cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def __len__(self,):
        return len()
    
    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('msa'):
            # print(x.shape)
            x = self.linear_embed_msa(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
            x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x
    
    
    

class SPTEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.linear_embed = nn.Linear(74, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, ta_perform):
        if ta_perform.startswith('msa'):
            x = self.linear_embed(x)
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
            x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, ta_perform=None):
        x = self.forward_features(x, ta_perform)
        return x


def noise_gen(is_train):
    min_snr, max_snr = -6, 18
    diff_snr = max_snr - min_snr
    
    min_var, max_var = 10**(-min_snr/20), 10**(-max_snr/20)
    diff_var = max_var - min_var
    if is_train:
        # b = torch.bernoulli(1/5.0*torch.ones(1))
        # if b > 0.5:
        #     channel_snr = torch.FloatTensor([20])
        # else:               
        #     channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        # noise_var = torch.rand(1)*diff_var+min_var  
        # channel_snr = 10*torch.log10((1/noise_var)**2)
        # channel_snr = torch.rand(1)*diff_snr+min_snr
        # noise_var = 10**(-channel_snr/20)
        channel_snr = torch.FloatTensor([12])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    else:
        channel_snr = torch.FloatTensor([12])
        noise_var = torch.FloatTensor([1]) * 10**(-channel_snr/20)  
    return channel_snr, noise_var 


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents, SNRdB, bit_per_index):
        latents = latents # [B x L x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BL x D]
        device = latents.device
        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BL x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BL, 1]
        shape = encoding_inds.shape
        Rx_signal = transmit(encoding_inds, SNRdB, bit_per_index)
        encoding_inds = torch.from_numpy(Rx_signal).to(device).reshape(shape)
        
        # Convert to one-hot encodings   
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BL x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BL, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x L x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()
        #print((quantized_latents - latents))
        return quantized_latents.contiguous(), vq_loss  # [B x L x D]
    

class Channels():
    def AWGN(self, Tx_sig, n_std):
        device = Tx_sig.device
        noise = torch.normal(0, n_std/math.sqrt(2), size=Tx_sig.shape).to(device)
        Rx_sig = Tx_sig + noise
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_std):
        device = Tx_sig.device
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_std, K=1):
        device = Tx_sig.device
        shape = Tx_sig.shape
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))
        H_real = torch.normal(mean, std, size=[1]).to(device)
        H_imag = torch.normal(mean, std, size=[1]).to(device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(device)
        Tx_sig = torch.matmul(Tx_sig.view(shape[0], -1, 2), H)
        Rx_sig = self.AWGN(Tx_sig, n_std)
        # Channel estimation
        Rx_sig = torch.matmul(Rx_sig, torch.inverse(H)).view(shape)

        return Rx_sig

