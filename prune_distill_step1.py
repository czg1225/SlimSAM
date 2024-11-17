import numpy as np
from torch.utils.data import DataLoader
import torch
from segment_anything_kd import sam_model_registry
from load_sam_json import SamDataset
from segment_anything_kd.utils.transforms import ResizeLongestSide
from prune_funcs import calculate_iou, get_pos_init, del_pos_init, prune_sam_step1
import json
from pycocotools import mask as mask_utils
import argparse

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description='SlimSAM')
parser.add_argument('--traindata_path', type=str,default = '')
parser.add_argument('--valdata_path', type=str,default = '')
parser.add_argument('--trainsize', type=int,default = 10000)
parser.add_argument('--gradsize', type=int,default = 1000)
parser.add_argument('--valsize', type=int,default = 50)
parser.add_argument('--epochs', type=int,default = 20)
parser.add_argument('--norm_type', type=str,default = 'mean')
parser.add_argument('--imptype', type=str,default = 'Disturb')
parser.add_argument('--global_way', type=bool,default = False)
parser.add_argument('--prune_ratio', type=float,default = 0.5)
args, unparsed = parser.parse_known_args()

def train_model():

    # torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    train_root_folder = args.traindata_path
    val_root_folder = args.valdata_path
    TRAIN_SIZE = args.trainsize
    VAL_SIZE = args.valsize
    GRAD_SIZE = args.gradsize
    num_train_epochs = args.epochs
    
    # Creating dataset loaders
    batch_size = 1

    grad_dataset = SamDataset(root_folder=train_root_folder, dataset_size=GRAD_SIZE, val=False)
    grad_loader = DataLoader(dataset=grad_dataset, batch_size=1, shuffle=False, num_workers=4,
                              pin_memory=True, drop_last=True)

    train_dataset = SamDataset(root_folder=train_root_folder, dataset_size=TRAIN_SIZE, val=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    val_dataset = SamDataset(root_folder=val_root_folder, dataset_size=VAL_SIZE, val=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
                             pin_memory=True, drop_last=False)

    # student model
    model_type = 'vit_b'
    checkpoint = 'checkpoints/sam_vit_b_qkv.pth'
    model = sam_model_registry[model_type](checkpoint=checkpoint)

    # teacher model
    teacher_model_type = 'vit_b'
    checkpoint = 'checkpoints/sam_vit_b_qkv.pth'
    teacher_model = sam_model_registry[teacher_model_type](checkpoint=checkpoint)
    teacher_model.to(device)
    teacher_model.eval()


    MSE_loss = torch.nn.MSELoss()
    lr = 1e-4
    ratio = args.prune_ratio
    loss_fn = torch.nn.MSELoss()
    transform = ResizeLongestSide(1024)

    norm_type = args.norm_type
    imptype = args.imptype
    global_way = args.global_way
    a_weight = 0.5
    round_to = model.image_encoder.num_heads

    print("===========================Parameter Settings===========================")

    print("Pruning Ratio:",ratio)
    print("VIT num_heads:",round_to)
    print("norm_type:",norm_type)
    print("imptype:",imptype)
    print("global:",global_way)
    print("learning rate:",lr)
    print("a_weight:",a_weight)
    print("round_to",round_to)
    print("TRAIN_SIZE",TRAIN_SIZE,"VAL_SIZE",VAL_SIZE, "GRAD_SIZE",GRAD_SIZE,"Epochs",num_train_epochs)

    model_name = teacher_model_type
    example_inputs = torch.randn(1, 3, 1024, 1024)



    for k in range(1):
############################################get initial grad for importance estimation############################################
        best_iou = 0
        model.to(device)
        model.image_encoder.train()
        grad_iter = iter(grad_loader)

        for i in range(len(grad_iter)):

            batch = next(grad_iter)
            input_image = batch["input_image"].to(device)

            with torch.no_grad():
                teacher_embedding, teacher_qkv_emb1, teacher_qkv_emb2, teacher_mid_emb,_ = teacher_model.image_encoder(input_image)
                teacher_embedding += torch.normal(mean=0,std=0.01,size=(1, 256, 64, 64)).to(device)  #Disturbed image embedding
                
            student_embedding, student_qkv_emb1, student_qkv_emb2, student_mid_emb,_= model.image_encoder(input_image)
            
            loss = loss_fn(teacher_embedding, student_embedding)
            loss.backward()

    
        #########################################################################################################
        print("===========================Pruning Start===========================")
        #Embedding Pruning
        model.cpu().eval()
        model = del_pos_init(model)
        model.image_encoder = prune_sam_step1(model=model.image_encoder, example_inputs=example_inputs, model_name=model_name, round_to=round_to, ratio=ratio, imptype = imptype, norm_type=norm_type, global_way=global_way)
        model = get_pos_init(model)
        model.to(device)
 
        model.image_encoder = torch.nn.DataParallel(model.image_encoder)
        model.image_encoder.train()

        optimizer = torch.optim.Adam(model.image_encoder.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5,patience=3,verbose=True)

        model.zero_grad()
        teacher_model.zero_grad()

        #Bottleneck Aligning
        for epoch in range(num_train_epochs):

            teacher_model.image_encoder = torch.nn.DataParallel(teacher_model.image_encoder)
            teacher_model.image_encoder.eval()

            torch.cuda.empty_cache()
            train_iter = iter(train_loader)

            for i in range(len(train_iter)):

                batch = next(train_iter)
                input_image = batch["input_image"].to(device)

                with torch.no_grad():
                    teacher_embedding,teacher_qkv_emb1,teacher_qkv_emb2,teacher_mid_emb,teacher_block_emb = teacher_model.image_encoder(input_image)

                student_embedding, student_qkv_emb1, student_qkv_emb2, student_mid_emb,student_block_emb = model.image_encoder(input_image)

                if epoch<10:
                    loss = (1-a_weight)*loss_fn(student_embedding, teacher_embedding)+a_weight*loss_fn(student_qkv_emb1, teacher_qkv_emb1)+a_weight*loss_fn(student_qkv_emb2, student_qkv_emb2)+a_weight*loss_fn(student_mid_emb, teacher_mid_emb)
                    loss.backward()
                else:
                    loss = loss_fn(student_embedding, teacher_embedding)
                    loss.backward()               

                #### batchsize×4 ####
                if i%4==3:
                    optimizer.step()
                    optimizer.zero_grad()
                
                #validation
                if i == len(train_iter)-1:
                    teacher_model.image_encoder = teacher_model.image_encoder.module
                    iou = 0
                    model.image_encoder.eval()
                    with torch.no_grad():
                        val_iter = iter(val_loader)
                        for j in range(len(val_iter)):
                            batch = next(val_iter)

                            input_image = batch["input_image"].to(device)
                            input_size = batch["input_size"]
                            original_image_size = batch["original_image_size"]

                            original_image_size[0] = original_image_size[0].numpy()[0]
                            original_image_size[1] = original_image_size[1].numpy()[0]
                            original_image_size = ([original_image_size[0],original_image_size[1]])

                            input_size[0] = input_size[0].numpy()[0]
                            input_size[1] = input_size[1].numpy()[0]
                            input_size = ([input_size[0],input_size[1]])

                            id = batch["id"]
                            annot = batch["annot"][0]
                            path = id[0]

                            with open(annot, encoding="utf-8") as f:
                                dict_data = json.load(f)
                                dict_data = dict_data["annotations"]
                                sub_count = 0
                                sub_iou = 0
                                for example in dict_data:

                                    sub_count += 1

                                    input_point = np.array(example['point_coords'])
                                    input_label = np.array([1])

                                    mask = mask_utils.decode(example["segmentation"])
                                    
                                    point_coords = transform.apply_coords(input_point, original_image_size)
                                    coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                                    labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
                                    coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                                    points = (coords_torch, labels_torch)

                                    # Model inference
                                    image_embedding,_,_,_,_ = model.image_encoder(input_image)
                                    sparse_embeddings, dense_embeddings = model.prompt_encoder(
                                        points=points,
                                        boxes=None,
                                        masks=None,
                                    )
                                    low_res_masks, iou_predictions = model.mask_decoder(
                                    image_embeddings=image_embedding,
                                    image_pe=model.prompt_encoder.get_dense_pe(),
                                    sparse_prompt_embeddings=sparse_embeddings,
                                    dense_prompt_embeddings=dense_embeddings,
                                    multimask_output=False,
                                    )

                                    student_masks = teacher_model.postprocess_masks(low_res_masks, input_size, original_image_size)
                                    student_masks = student_masks > teacher_model.mask_threshold
                                    student_masks = student_masks[0].detach().cpu().numpy()[0]
                                    sub_iou += calculate_iou(student_masks, mask)
                                
                            sub_iou = sub_iou/sub_count
                            iou += sub_iou

                    
                    iou = iou/len(val_iter)
                    model.image_encoder.train()

                    model.image_encoder.eval()
                    if iou>=best_iou:
                        best_iou = iou
                        filename = 'checkpoints/vit_b_slim_step1_'+'.pth'
                        torch.save(model, filename)   
                        print("save checkpoint")
                    model.image_encoder.train()

                    scheduler.step(iou)
                    print("epoch:",epoch)
                    print("IOU: {} Best IOU {}".format(iou,best_iou))

        
        




        


        


    






           


            











if __name__ == '__main__':
    train_model()