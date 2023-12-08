import numpy as np
import torch
from segment_anything_kd import SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
import torch.nn as nn
import torch_pruning as tp

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))



# student model
model_type = 'vit_b'
checkpoint = None
student_model = sam_model_registry[model_type](checkpoint=checkpoint)
print(student_model.image_encoder)

state_dict = torch.load('checkpoints/sam_vit_b_0b3195.pth')

for i in range(0,24):
    originkey = "image_encoder.blocks."+str(i)+".attn.qkv.weight"
    state_dict["image_encoder.blocks."+str(i)+".attn.q.weight"] = state_dict[originkey][0:1024]
    state_dict["image_encoder.blocks."+str(i)+".attn.k.weight"] = state_dict[originkey][1024:2048]
    state_dict["image_encoder.blocks."+str(i)+".attn.v.weight"] = state_dict[originkey][2048:3072]
    del state_dict[originkey]

    originkey = "image_encoder.blocks."+str(i)+".attn.qkv.bias"
    state_dict["image_encoder.blocks."+str(i)+".attn.q.bias"] = state_dict[originkey][0:1024]
    state_dict["image_encoder.blocks."+str(i)+".attn.k.bias"] = state_dict[originkey][1024:2048]
    state_dict["image_encoder.blocks."+str(i)+".attn.v.bias"] = state_dict[originkey][2048:3072]
    del state_dict[originkey]


for key, value in state_dict.items():
    print(key,value.shape)

torch.save(state_dict,'checkpoints/sam_vit_b_qkv.pth')



# model_path = "checkpoints/SlimSAM-77.pth"
# SlimSAM_model = torch.load(model_path)
# SlimSAM_model.image_encoder = SlimSAM_model.image_encoder.module
# SlimSAM_model.to(device)
# SlimSAM_model.eval()
# print("model_path:",model_path)

# def forward(self, x):

#     x = self.patch_embed(x)
#     if self.pos_embed is not None:
#         x = x + self.pos_embed

#     for blk in self.blocks:
#         x,qkv_emb,mid_emb,x_emb = blk(x)

#     x = self.neck(x.permute(0, 3, 1, 2))
    
#     return x

# import types
# funcType = types.MethodType
# SlimSAM_model.image_encoder.forward = funcType(forward, SlimSAM_model.image_encoder)

# depth = SlimSAM_model.image_encoder.depth

# state_dict = SlimSAM_model.state_dict()
# for i in range(depth):
#     qw = state_dict["image_encoder.blocks."+str(i)+".attn.q.weight"]
#     kw = state_dict["image_encoder.blocks."+str(i)+".attn.k.weight"]
#     vw = state_dict["image_encoder.blocks."+str(i)+".attn.v.weight"]

#     qb = state_dict["image_encoder.blocks."+str(i)+".attn.q.bias"]
#     kb = state_dict["image_encoder.blocks."+str(i)+".attn.k.bias"]
#     vb = state_dict["image_encoder.blocks."+str(i)+".attn.v.bias"]

#     SlimSAM_model.image_encoder.blocks[i].attn.qkv = nn.Linear(qw.shape[1], qw.shape[0]*3, bias=True)
#     SlimSAM_model.image_encoder.blocks[i].attn.qkv.weight = nn.Parameter(torch.cat([qw,kw,vw],dim=0))
#     SlimSAM_model.image_encoder.blocks[i].attn.qkv.bias = nn.Parameter(torch.cat([qb,kb,vb],dim=0))

#     del SlimSAM_model.image_encoder.blocks[i].attn.q
#     del SlimSAM_model.image_encoder.blocks[i].attn.k
#     del SlimSAM_model.image_encoder.blocks[i].attn.v



#     def forward(self, x: torch.Tensor) -> torch.Tensor:

#         B, H, W, _ = x.shape

#         # qkv with shape (3, B, nHead, H * W, C)
#         qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         # q, k, v with shape (B * nHead, H * W, C)
#         q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

#         qkv_emb = torch.cat([q,k,v],dim=0)

#         attn = (q * self.scale) @ k.transpose(-2, -1)

#         if self.use_rel_pos:
#             attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)

#         x = self.proj(x)

#         return x, qkv_emb

# import types
# funcType = types.MethodType
# for i in range(depth):
#     SlimSAM_model.image_encoder.blocks[i].attn.forward = funcType(forward, SlimSAM_model.image_encoder.blocks[i].attn)

# print(SlimSAM_model.image_encoder)

# example_inputs = torch.randn(1, 3, 1024, 1024).to(device)
# ori_macs, ori_size = tp.utils.count_ops_and_params(SlimSAM_model.image_encoder, example_inputs)
# print("MACs(G):",ori_macs/1e9,"Params(M):",ori_size/1e6)

