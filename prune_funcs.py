import numpy as np
import torch
import torch.nn as nn
from segment_anything_kd.modeling.image_encoder import Attention
from segment_anything_kd.modeling.common import LayerNorm2d
import torch_pruning as tp


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) for two binary masks.

    Parameters:
        mask1 (numpy.ndarray): The first binary mask.
        mask2 (numpy.ndarray): The second binary mask.

    Returns:
        float: The IoU score.
    """
    # Make sure the input masks have the same shape
    assert mask1.shape == mask2.shape, "Both masks must have the same shape."

    # Calculate the intersection and union of the masks
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Compute the IoU score
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def get_pos_init(model):
    depth = model.image_encoder.depth
    for i in range(depth):
        head_dim = model.image_encoder.blocks[i].attn.q.out_features // model.image_encoder.blocks[i].attn.num_heads
        input_size = model.image_encoder.blocks[i].attn.input_size
        model.image_encoder.blocks[i].attn.scale = head_dim**-0.5
        model.image_encoder.blocks[i].attn.use_rel_pos = True
        model.image_encoder.blocks[i].attn.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
        model.image_encoder.blocks[i].attn.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    return model


def del_pos_init(model):
    depth = model.image_encoder.depth
    for i in range(depth):
        model.image_encoder.blocks[i].attn.use_rel_pos = False
        del(model.image_encoder.blocks[i].attn.rel_pos_h) 
        del(model.image_encoder.blocks[i].attn.rel_pos_w) 
    return model



def prune_sam_step1(model, example_inputs, model_name, round_to, ratio,imptype,norm_type,global_way):

    ignored_layers = []

    #########################################
    # Ignore unprunable modules
    #########################################
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == 256:
            ignored_layers.append(m) 
        if isinstance(m, LayerNorm2d):
            ignored_layers.append(m)
    
    for n in range(12):
        ignored_layers.append(model.blocks[n].attn.q)
        ignored_layers.append(model.blocks[n].attn.k)
        ignored_layers.append(model.blocks[n].attn.v)
        ignored_layers.append(model.blocks[n].mlp.lin1)

    # print(ignored_layers)
    # For ViT: Rounding the number of channels to the nearest multiple of num_heads
    round_to = round_to

    #########################################
    # (Optional) Register unwrapped nn.Parameters 
    # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
    # If you want to prune other dims, you can register them here.
    #########################################
    unwrapped_parameters = None

    #########################################
    # Build network pruners
    #########################################

    if imptype == "Disturb":
        importance = tp.importance.DisturbImportance(normalizer=norm_type ,group_reduction="mean")
    elif imptype == "mag":
        importance = tp.importance.MagnitudeImportance(p=2, normalizer=norm_type, group_reduction="mean")
    elif imptype == "taylor":
        importance = tp.importance.TaylorImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "random":
        importance = tp.importance.RandomImportance()


    channel_groups = {}
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, Attention):
            channel_groups[m.q] = m.num_heads
            channel_groups[m.k] = m.num_heads
            channel_groups[m.v] = m.num_heads

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        ch_sparsity=ratio,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        global_pruning=global_way,
        channel_groups=channel_groups,
    )

    #########################################
    # Pruning 
    #########################################

    for i in range(iterative_steps):

        ori_macs, ori_size = tp.utils.count_ops_and_params(model, example_inputs)

        pruner.step()
        #########################################
        # Testing 
        #########################################
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            print("{} Pruning: ".format(model_name))
            macs_after_prune, params_after_prune = tp.utils.count_ops_and_params(model, example_inputs)
            print("  Params: %s => %s" % (ori_size, params_after_prune))
            print("  Macs: %s => %s" % (ori_macs, macs_after_prune))

            if isinstance(out, (dict,list,tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")
    
    return model



def prune_sam_step2_local(model, example_inputs, model_name, round_to, ratio,imptype,norm_type,global_way):

    ignored_layers = []

    #########################################
    # Ignore unprunable modules
    #########################################
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            ignored_layers.append(m) 
        if isinstance(m, LayerNorm2d):
            ignored_layers.append(m)

    # print(ignored_layers)
    # For ViT: Rounding the number of channels to the nearest multiple of num_heads
    round_to = round_to

    #########################################
    # (Optional) Register unwrapped nn.Parameters 
    # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
    # If you want to prune other dims, you can register them here.
    #########################################
    unwrapped_parameters = None

    #########################################
    # Build network pruners
    #########################################

    if imptype == "Disturb":
        importance = tp.importance.DisturbImportance(normalizer=norm_type ,group_reduction="mean")
    elif imptype == "mag":
        importance = tp.importance.MagnitudeImportance(p=2, normalizer=norm_type, group_reduction="mean")
    elif imptype == "taylor":
        importance = tp.importance.TaylorImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "random":
        importance = tp.importance.RandomImportance()


    channel_groups = {}
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, Attention):
            channel_groups[m.q] = m.num_heads
            channel_groups[m.k] = m.num_heads
            channel_groups[m.v] = m.num_heads

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        ch_sparsity=ratio,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        global_pruning=global_way,
        channel_groups=channel_groups,
    )

    #########################################
    # Pruning 
    #########################################

    for i in range(iterative_steps):

        ori_macs, ori_size = tp.utils.count_ops_and_params(model, example_inputs)

        pruner.step()
        #########################################
        # Testing 
        #########################################
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            print("{} Pruning: ".format(model_name))
            macs_after_prune, params_after_prune = tp.utils.count_ops_and_params(model, example_inputs)
            print("  Params: %s => %s" % (ori_size, params_after_prune))
            print("  Macs: %s => %s" % (ori_macs, macs_after_prune))

            if isinstance(out, (dict,list,tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")
    
    return model





def prune_sam_step2_global(model, example_inputs, model_name, round_to, ratio,imptype,norm_type,global_way,gs):

    ignored_layers = []
    #########################################
    # Ignore unprunable modules
    #########################################
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            ignored_layers.append(m) 
        if isinstance(m, LayerNorm2d):
            ignored_layers.append(m)
            
    if gs == 1:
        for n in range(12):
            ignored_layers.append(model.blocks[n].mlp.lin1)
    if gs == 2:
        for n in range(12):
            ignored_layers.append(model.blocks[n].attn.q)
            ignored_layers.append(model.blocks[n].attn.k)
            ignored_layers.append(model.blocks[n].attn.v)


    # print(ignored_layers)
    # For ViT: Rounding the number of channels to the nearest multiple of num_heads
    round_to = round_to

    #########################################
    # (Optional) Register unwrapped nn.Parameters 
    # TP will automatically detect unwrapped parameters and prune the last dim for you by default.
    # If you want to prune other dims, you can register them here.
    #########################################
    unwrapped_parameters = None

    #########################################
    # Build network pruners
    #########################################
    if imptype == "Disturb":
        importance = tp.importance.DisturbImportance(normalizer=norm_type ,group_reduction="mean")
    elif imptype == "mag":
        importance = tp.importance.MagnitudeImportance(p=2, normalizer=norm_type, group_reduction="mean")
    elif imptype == "taylor":
        importance = tp.importance.TaylorImportance(normalizer=norm_type, group_reduction="mean")
    elif imptype == "random":
        importance = tp.importance.RandomImportance()


    channel_groups = {}
    # All heads should be pruned simultaneously, so we group channels by head.
    for m in model.modules():
        if isinstance(m, Attention):
            channel_groups[m.q] = m.num_heads
            channel_groups[m.k] = m.num_heads
            channel_groups[m.v] = m.num_heads

    iterative_steps = 1
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        ch_sparsity=ratio,
        round_to=round_to,
        unwrapped_parameters=unwrapped_parameters,
        ignored_layers=ignored_layers,
        global_pruning=global_way,
        channel_groups=channel_groups,
    )

    #########################################
    # Pruning 
    #########################################

    for i in range(iterative_steps):

        ori_macs, ori_size = tp.utils.count_ops_and_params(model, example_inputs)

        pruner.step()
        #########################################
        # Testing 
        #########################################
        with torch.no_grad():
            if isinstance(example_inputs, dict):
                out = model(**example_inputs)
            else:
                out = model(example_inputs)
            print("{} Pruning: ".format(model_name))
            macs_after_prune, params_after_prune = tp.utils.count_ops_and_params(model, example_inputs)
            print("  Params: %s => %s" % (ori_size, params_after_prune))
            print("  Macs: %s => %s" % (ori_macs, macs_after_prune))

            if isinstance(out, (dict,list,tuple)):
                print("  Output:")
                for o in tp.utils.flatten_as_list(out):
                    print(o.shape)
            else:
                print("  Output:", out.shape)
            print("------------------------------------------------------\n")
    
    return model


