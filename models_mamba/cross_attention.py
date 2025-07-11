import torch
import torch.nn as nn
import torch.nn.functional as F

def split_image_tensor(input_tensor, block_size):
    
    # 获取输入张量的形状
    batch_size, channels, height, width = input_tensor.shape
    
    # 计算沿高度和宽度方向的小块数量
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size
    
    # 使用 unfold 将每个通道切成指定大小的小块
    unfolded_tensor = input_tensor.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    
    # 调整维度顺序并合并批次和块数
    unfolded_tensor = unfolded_tensor.permute(0, 2, 3, 1, 4, 5).contiguous()
    
    # 最后调整形状
    output_tensor = unfolded_tensor.view(-1, channels, block_size, block_size)
    
    return output_tensor

def combine_image_tensor(input_tensor, block_size, original_height, original_width):
    batch_size = input_tensor.size(0) // ((original_height // block_size) * (original_width // block_size))
    channels = input_tensor.size(1)

    num_blocks_height = original_height // block_size
    num_blocks_width = original_width // block_size
    input_tensor = input_tensor.view(batch_size, num_blocks_height, num_blocks_width, channels, block_size, block_size)
    
    input_tensor = input_tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
    
    output_tensor = input_tensor.view(batch_size, channels, original_height, original_width)
    
    return output_tensor

class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5
 
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads
 
 
        self.embedding = nn.Linear(6, emb_dim) #改
        self.embedding_2 = nn.Linear(12, emb_dim) #改
        self.down_conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=4, stride=4, padding=0)
 
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
 
        self.proj_out = nn.Conv2d(emb_dim*2, in_channels, kernel_size=1, stride=1, padding=0)
        self.deconv_layer = nn.ConvTranspose2d(in_channels=emb_dim*2, out_channels=6, kernel_size=4, stride=4, padding=0)
 
 
    def forward(self, img1, img2, pad_mask=None):
        # img1 = self.down_conv(img1)
        # img2 = self.down_conv(img2)
        img1 = split_image_tensor(img1,16)
        img2 = split_image_tensor(img2,16)
        B, C, H, W = img1.shape  # 假设输入形状为 [B, 6, 256, 256]
        img1_flat = img1.view(B, C, H * W).permute(0, 2, 1)  # (B, 16*16, 6)
        img2_flat = img2.view(B, C, H * W).permute(0, 2, 1)  # (B, 16*16, 6)

        img_cat = torch.cat((img1_flat, img2_flat), dim=2)

        # 将 6 维通道特征嵌入到指定的 embed_dim 维度
        img1_emb = self.embedding(img1_flat)  # (B, 16*16, embed_dim)
        img2_emb = self.embedding(img2_flat)
        img_emb = self.embedding_2(img_cat)
 
        Q1 = self.Wq(img1_emb)  # [batch_size, h*w, emb_dim] = (B, 16*16, embed_dim)
        Q2 = self.Wq(img2_emb)
        K = self.Wk(img_emb)  # [batch_szie, seq_len, emb_dim] = (B, 16*16, embed_dim)
        V = self.Wv(img_emb)
 
        Q1 = Q1.view(B, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        Q2 = Q2.view(B, -1, self.num_heads, self.depth).transpose(1, 2) 
        K = K.view(B, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(B, -1, self.num_heads, self.depth).transpose(1, 2)
 
        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q1, K)
        att_weights = att_weights * self.scale 
 
        # if pad_mask is not None:
        #     # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
        #     pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        #     att_weights = att_weights.masked_fill(pad_mask, -1e9)
 
        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)

        att_weights1 = torch.einsum('bnid,bnjd -> bnij', Q2, K)
        att_weights1 = att_weights1 * self.scale
        att_weights1 = F.softmax(att_weights1, dim=-1)
        out1 = torch.einsum('bnij, bnjd -> bnid', att_weights1, V) 

        out = out.transpose(1, 2).contiguous().view(B, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]
        out = out.permute(0, 2, 1).view(B, self.emb_dim, H, W) 

        out1 = out1.transpose(1, 2).contiguous().view(B, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]
        out1 = out1.permute(0, 2, 1).view(B, self.emb_dim, H, W)
        out = torch.cat((out, out1), dim=1) 
        #out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = self.proj_out(out)   # [batch_size, c, h, w]
        #out = self.deconv_layer(out)
        out = combine_image_tensor(out, 16, 128, 128)
        return out