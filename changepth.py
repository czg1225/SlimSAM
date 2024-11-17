import numpy as np
import torch

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))



# # student model
# model_type = 'vit_b'
# checkpoint = None
# student_model = sam_model_registry[model_type](checkpoint=checkpoint)
# print(student_model.image_encoder)

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


