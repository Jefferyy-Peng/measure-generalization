import os.path
from collections import OrderedDict

import timm
import torch
from transformers import ViTForImageClassification, ViTConfig

from model_zoo.utils import compare_model_outputs
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.models.layers.head import Head
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from model_zoo.utils import convert_weights

model_dict = {
    'ViT-B_16-in21k':'vit_base_patch16_224',
    'ViT-B_16-clip-openai': 'open-clip:timm/vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-openai-0.03': 'open-clip:timm/vit_base_patch16_clip_224.openai',
    'ViT-B_16-clip-laion2b': 'open-clip:timm/vit_base_patch16_clip_224.laion400m_e31',
    'ViT-B_16-clip-laion2b-0.03': 'open-clip:timm/vit_base_patch16_clip_224.laion400m_e31',
    'ViT-B_16-mae': 'vit_base_patch16_224',
    'ViT-B_16-dinov2': 'facebook/dino-vitb16',
    'ViT-B_16-in1k': 'vit_base_patch16_224',
    'ViT-B_16-in21k-lp':'vit_base_patch16_224',
    'ViT-B_16-clip-openai-lp': 'openai/clip-vit-base-patch16',
    'ViT-B_16-clip-laion2b-lp': 'open-clip:timm/vit_base_patch16_clip_224.laion400m_e31',
    'ViT-B_16-mae-lp': 'vit_base_patch16_224',
    'ViT-B_16-dinov2-lp': 'facebook/dino-vitb16',
    'ViT-B_16-in1k-lp': 'vit_base_patch16_224',
    'ViT-B_16-scratch': 'vit_base_patch16_224',
}

def convert_hf_to_hookedvit(hf_sd, config):
    vit_sd = OrderedDict()
    num_heads = config.n_heads
    head_dim = config.d_head

    for k, v in hf_sd.items():
        if k == "vit.embeddings.cls_token":
            vit_sd["cls_token"] = v
        elif k == "vit.embeddings.position_embeddings":
            vit_sd["pos_embed.W_pos"] = v.squeeze(0)
        elif k == "vit.embeddings.patch_embeddings.projection.weight":
            vit_sd["embed.proj.weight"] = v
        elif k == "vit.embeddings.patch_embeddings.projection.bias":
            vit_sd["embed.proj.bias"] = v

        elif k.startswith("vit.encoder.layer."):
            parts = k.split(".")
            layer = int(parts[3])
            sub = ".".join(parts[4:])

            prefix = f"blocks.{layer}"

            if sub == "layernorm_before.weight":
                vit_sd[f"{prefix}.ln1.w"] = v
            elif sub == "layernorm_before.bias":
                vit_sd[f"{prefix}.ln1.b"] = v
            elif sub == "layernorm_after.weight":
                vit_sd[f"{prefix}.ln2.w"] = v
            elif sub == "layernorm_after.bias":
                vit_sd[f"{prefix}.ln2.b"] = v

            elif sub == "attention.attention.query.weight":
                vit_sd[f"{prefix}.attn.W_Q"] = v.T.reshape(head_dim * num_heads, num_heads, head_dim).permute(1, 0, 2).contiguous()
            elif sub == "attention.attention.key.weight":
                vit_sd[f"{prefix}.attn.W_K"] = v.T.reshape(head_dim * num_heads, num_heads, head_dim).permute(1, 0, 2).contiguous()
            elif sub == "attention.attention.value.weight":
                vit_sd[f"{prefix}.attn.W_V"] = v.T.reshape(head_dim * num_heads, num_heads, head_dim).permute(1, 0, 2).contiguous()
            elif sub == "attention.output.dense.weight":
                vit_sd[f"{prefix}.attn.W_O"] = v.T.reshape(num_heads, head_dim, head_dim * num_heads).contiguous()

            elif sub == "attention.attention.query.bias":
                vit_sd[f"{prefix}.attn.b_Q"] = v.view(num_heads, head_dim)
            elif sub == "attention.attention.key.bias":
                vit_sd[f"{prefix}.attn.b_K"] = v.view(num_heads, head_dim)
            elif sub == "attention.attention.value.bias":
                vit_sd[f"{prefix}.attn.b_V"] = v.view(num_heads, head_dim)
            elif sub == "attention.output.dense.bias":
                vit_sd[f"{prefix}.attn.b_O"] = v

            elif sub == "intermediate.dense.weight":
                vit_sd[f"{prefix}.mlp.W_in"] = v.T  # Transpose!
            elif sub == "intermediate.dense.bias":
                vit_sd[f"{prefix}.mlp.b_in"] = v
            elif sub == "output.dense.weight":
                vit_sd[f"{prefix}.mlp.W_out"] = v.T  # Transpose!
            elif sub == "output.dense.bias":
                vit_sd[f"{prefix}.mlp.b_out"] = v

        elif k == "vit.layernorm.weight":
            vit_sd["ln_final.w"] = v
        elif k == "vit.layernorm.bias":
            vit_sd["ln_final.b"] = v
        elif k == "classifier.weight":
            vit_sd["head.W_H"] = v.T  # Transpose!
        elif k == "classifier.bias":
            vit_sd["head.b_H"] = v
    return vit_sd

def get_model_from_ckpt(model_name, dataset_name, ckpt_path, ckpt_root_path, device='cuda:0'):
    sd_name = model_dict[model_name]
    total_ckpt_path = os.path.join(ckpt_root_path, ckpt_path)
    state_dict = torch.load(total_ckpt_path, map_location=device)
    model = HookedViT.from_pretrained(
        sd_name,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        allow_failing=True,
    )
    model.cfg.device = device
    model.cfg.normalize_output = False
    setattr(model.cfg, "use_normalization_before_and_after", False)
    config = model.cfg
    config.n_classes = 2
    model.head = Head(config)
    converted_weights = convert_weights(state_dict, model_name, config)
    missing, unexpected = model.load_state_dict(converted_weights, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    # original_model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True,
    #                                    num_classes=2,
    #                                    drop_rate=0.1,
    #                                    img_size=224)
    # original_model.load_state_dict(state_dict)
    # compare_model_outputs(model, original_model)
    model = model.to(device)
    return model
