""" Full assembly of the parts to form the complete network """

from scripts.Unet_parts import *
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
import math
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class RLN(nn.Module):
	r"""Revised LayerNorm"""
	def __init__(self, dim, eps=1e-5, detach_grad=False):
		super(RLN, self).__init__()
		self.eps = eps
		self.detach_grad = detach_grad

		self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
		self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

		self.meta1 = nn.Conv2d(1, dim, 1)
		self.meta2 = nn.Conv2d(1, dim, 1)

		trunc_normal_(self.meta1.weight, std=.02)
		nn.init.constant_(self.meta1.bias, 1)

		trunc_normal_(self.meta2.weight, std=.02)
		nn.init.constant_(self.meta2.bias, 0)

	def forward(self, input):
		mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
		std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

		normalized_input = (input - mean) / std

		if self.detach_grad:
			rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
		else:
			rescale, rebias = self.meta1(std), self.meta2(mean)

		out = normalized_input * self.weight + self.bias
		return out, rescale, rebias

class Mlp(nn.Module):
	def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
		super().__init__()
		out_features = out_features or in_features
		hidden_features = hidden_features or in_features

		self.network_depth = network_depth

		self.mlp = nn.Sequential(
			nn.Conv2d(in_features, hidden_features, 1),
			nn.ReLU(True),
			nn.Conv2d(hidden_features, out_features, 1)
		)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			gain = (8 * self.network_depth) ** (-1/4)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
			std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
			trunc_normal_(m.weight, std=std)
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		return self.mlp(x)


def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)
	return windows


def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x


def get_relative_positions(window_size):
	coords_h = torch.arange(window_size)
	coords_w = torch.arange(window_size)

	coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
	coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
	relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

	relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
	relative_positions_log  = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

	return relative_positions_log



class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		head_dim = dim // num_heads
		self.scale = head_dim ** -0.5

		relative_positions = get_relative_positions(self.window_size)
		self.register_buffer("relative_positions", relative_positions)
		self.meta = nn.Sequential(
			nn.Linear(2, 256, bias=True),
			nn.ReLU(True),
			nn.Linear(256, num_heads, bias=True)
		)

		self.softmax = nn.Softmax(dim=-1)

	def forward(self, qkv):
		B_, N, _ = qkv.shape

		qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

		q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

		q = q * self.scale
		attn = (q @ k.transpose(-2, -1))

		relative_position_bias = self.meta(self.relative_positions)
		relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
		attn = attn + relative_position_bias.unsqueeze(0)

		attn = self.softmax(attn)

		x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
		return x


class Attention(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size

		self.network_depth = network_depth
		self.use_attn = use_attn
		self.conv_type = conv_type

		if self.conv_type == 'Conv':
			self.conv = nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
				nn.ReLU(True),
				nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
			)

		if self.conv_type == 'DWConv':
			self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

		if self.conv_type == 'DWConv' or self.use_attn:
			self.V = nn.Conv2d(dim, dim, 1)
			self.proj = nn.Conv2d(dim, dim, 1)

		if self.use_attn:
			self.QK = nn.Conv2d(dim, dim * 2, 1)
			self.attn = WindowAttention(dim, window_size, num_heads)

		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Conv2d):
			w_shape = m.weight.shape
			
			if w_shape[0] == self.dim * 2:	# QK
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)		
			else:
				gain = (8 * self.network_depth) ** (-1/4)
				fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
				std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
				trunc_normal_(m.weight, std=std)

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)

	def check_size(self, x, shift=False):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape

		if self.conv_type == 'DWConv' or self.use_attn:
			V = self.V(X)

		if self.use_attn:
			QK = self.QK(X)
			QKV = torch.cat([QK, V], dim=1)

			# shift
			shifted_QKV = self.check_size(QKV, self.shift_size > 0)
			Ht, Wt = shifted_QKV.shape[2:]

			# partition windows
			shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
			qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C

			attn_windows = self.attn(qkv)

			# merge windows
			shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

			# reverse cyclic shift
			out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
			attn_out = out.permute(0, 3, 1, 2)

			if self.conv_type in ['Conv', 'DWConv']:
				conv_out = self.conv(V)
				out = self.proj(conv_out + attn_out)
			else:
				out = self.proj(attn_out)

		else:
			if self.conv_type == 'Conv':
				out = self.conv(X)				# no attention and use conv, no projection
			elif self.conv_type == 'DWConv':
				out = self.proj(self.conv(V))

		return out


class TransformerBlock(nn.Module):
	def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
				 norm_layer=nn.LayerNorm, mlp_norm=False,
				 window_size=8, shift_size=0, use_attn=True, conv_type=None):
		super().__init__()
		self.use_attn = use_attn
		self.mlp_norm = mlp_norm

		self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
		self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
							  shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

		self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
		self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

	def forward(self, x):
		identity = x
		if self.use_attn: x, rescale, rebias = self.norm1(x)
		x = self.attn(x)
		if self.use_attn: x = x * rescale + rebias
		x = identity + x

		identity = x
		if self.use_attn and self.mlp_norm: x, rescale, rebias = self.norm2(x)
		x = self.mlp(x)
		if self.use_attn and self.mlp_norm: x = x * rescale + rebias
		x = identity + x
		return x

class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()
		
		self.height = height
		d = max(int(dim/reduction), 4)
		
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(dim, d, 1, bias=False), 
			nn.ReLU(),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)
		
		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		
		B, C, H, W = in_feats[0].shape
		
		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)
		
		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(self.avg_pool(feats_sum))
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out      


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
            super(SEBlock, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels, bias=False),
                nn.Sigmoid()
            )
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class PatchEmbed(nn.Module):
	def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
		super().__init__()
		self.in_chans = in_chans
		self.embed_dim = embed_dim

		if kernel_size is None:
			kernel_size = patch_size

		self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

	def forward(self, x):
		x = self.proj(x)
		return x
class ConvNet(nn.Module):
    def __init__(self,inchannel,outchanel):
        super(ConvNet, self).__init__()
        # input shape: (16, 24, 256, 256)
        self.conv1 = nn.Conv2d(inchannel, inchannel*4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(inchannel*4, outchanel*2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(outchanel*2, outchanel, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # input shape: (16, 24, 256, 256)
        x = self.relu(self.conv1(x))  # shape: (16, 12, 256, 256)
        x = self.relu(self.conv2(x))  # shape: (16, 6, 256, 256)
        x = self.relu(self.conv3(x))  # shape: (16, 3, 256, 256)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,depth = 6,window_size = 8,attn_loc='last',attn_ratio = 0, bilinear=False,embed_dims=[24, 48, 96, 48, 24],norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # ## ir
        # attn_depth = attn_ratio * depth
        # if attn_loc == 'last':
        #     use_attns = [i >= depth-attn_depth for i in range(depth)]

        # self.patch_embed = PatchEmbed(patch_size=1, in_chans=n_channels, embed_dim=embed_dims[0], kernel_size=3)
        # num_heads = 8
        # self.layers = nn.ModuleList([TransformerBlock(network_depth=6,
		# 					 dim=embed_dims[0], 
		# 					 num_heads=num_heads,
		# 					 mlp_ratio=4,
		# 					 norm_layer=norm_layer,
		# 					 window_size=window_size,
		# 					 shift_size=0 if (i % 2 == 0) else window_size // 2,
        # 					 use_attn=use_attns[i], conv_type='DWConv') for i in range(0,depth)])
        # self.Convnet = ConvNet(24,3)
        ## ir
        self.inc2 = (DoubleConv(n_channels-1, 64))
        self.se12 = SEBlock(64)
        # self.down12 = (Down(64, 128))
        # self.se22 = SEBlock(128)
        # self.down22 = (Down(128, 256))
        # self.se32 = SEBlock(256)
        # self.down32 = (Down(256, 512))
        # self.se42 = SEBlock(512)
        # factor = 2 if bilinear else 1
        # self.down42 = (Down(512, 1024 // factor))
        # self.se52 = SEBlock(1024)
        
        ## rgb
        self.inc = (DoubleConv(n_channels, 64))
        self.se1 = SEBlock(64)
        self.conv1 = ConvNet(128,64)
        self.se10 = SEBlock(64)
        
        self.down1 = (Down(64, 128))
        self.se2 = SEBlock(128)
        self.down2 = (Down(128, 256))
        self.se3 = SEBlock(256)
        self.down3 = (Down(256, 512))
        self.se4 = SEBlock(512)
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.se5 = SEBlock(1024)

        # self.skF2 = SKFusion(1024)
        ## 反卷积
        
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.se6 = SEBlock(512)
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.se7 = SEBlock(256)
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.se8 = SEBlock(128)
        self.up4 = (Up(128, 64, bilinear))
        self.se9 = SEBlock(64)

        self.outc = (OutConv(64, 3))
        

    def forward(self, x,ir):
        # init = x.clone()
        # ir = self.patch_embed(ir)
        # for layer in self.layers:
        #     ir = layer(ir)
        # ir = self.Convnet(ir)
        # new = self.skF1([x, ir]) + x
        # new = torch.cat([x, ir], dim=1)
        # rgb
        # ir
        x12 = self.inc2(ir)
        x12 = self.se1(x12)
                ## 融合

        
        x1 = self.inc(x)
        x1 = self.se1(x1)
        new = torch.cat([x1,x12], dim=1)
        # x5 = self.skF2([x5, x52]) + x5
        new = self.conv1(new)
        new = self.se10(new)
        x2 = self.down1(new)
        x2 = self.se2(x2)
        x3 = self.down2(x2)
        x3 = self.se3(x3)
        x4 = self.down3(x3)
        x4 = self.se4(x4)
        x5 = self.down4(x4)
        x5 = self.se5(x5)

        # x22 = self.down1(x12)
        # x22 = self.se2(x22)
        # x32 = self.down2(x22)
        # x32 = self.se3(x32)
        # x42 = self.down3(x32)
        # x42 = self.se4(x42)
        # x52 = self.down4(x42)
        # x52 = self.se5(x52)

        ## 反卷积
        x = self.up1(x5 , x4)
        x = self.se6(x)
        x = self.up2(x, x3)
        x = self.se7(x)
        x = self.up3(x, x2)
        x = self.se8(x)
        x = self.up4(x, x1)
        x = self.se9(x)
        # x = torch.cat([x, init], dim=1)
        logits = self.outc(x)
        # x = self.Convnet4(x)
        # logits = self.patch_embed3(x)
        # for layer in self.layers3:
        #     logits = layer(logits)
        
        return logits