# contextvla_model.py
import os

from huggingface_hub import snapshot_download

from transformers.modeling_utils import load_sharded_checkpoint
from transformers import AutoConfig

from src.models.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from torch import nn
import torch
from src.models.layer_wrapper import LayerWrapper


class IndexContext:
    batch_indices: int
    gather_indices: int


class ContextVLA_Qwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):

        base_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        model = Qwen2_5_VLForConditionalGeneration._from_config(base_config, **kwargs)

        index_context = IndexContext()
        for layer_idx in range(len(model.model.layers)):
            model.model.layers[layer_idx] = LayerWrapper(
                model.model.layers[layer_idx],
                layer_idx=layer_idx,
                internal_projection=2,
                num_frames=8,
                num_views=3,
                index_context=index_context,
                img_pattern=[151652],
                motion_token=1,
            )

        # expand vocab
        old_weight = model.model.embed_tokens.weight.data
        new_embedding = nn.Embedding(153713, old_weight.shape[1])
        with torch.no_grad():
            new_embedding.weight[:151664].copy_(old_weight[:151664])
        model.model.embed_tokens = new_embedding

        old_head = model.lm_head
        new_head = nn.Linear(old_head.weight.data.shape[1], 153713, bias=False)
        with torch.no_grad():
            new_head.weight[:151664].copy_(old_head.weight[:151664])
        model.lm_head = new_head
        model.vocab_size = model.config.vocab_size = 153713

        if os.path.isdir(pretrained_model_name_or_path):
            local_dir = pretrained_model_name_or_path
        else:
            local_dir = snapshot_download(pretrained_model_name_or_path)

        load_sharded_checkpoint(model, local_dir)
        print(f"[ContextVLA] weights loaded from {local_dir}")
        
        return model
    
