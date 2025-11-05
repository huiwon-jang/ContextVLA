import numpy as np
from openpi_client import base_policy as _base_policy
from src.models import layer_wrapper, vision_process
import torch
import torch.nn as nn
import json

from transformers import AutoProcessor
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
from PIL import Image

class IndexContext:
    batch_indices: int
    gather_indices: int

ACTION_TOKEN_MIN = 151665
ACTION_TOKEN_MAX = 153712


def arrays_equal_values(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, equal_nan=True, atol=0)


class ContextVLAPolicy(_base_policy.BasePolicy):
    def __init__(
            self, 
            ckpt_path, 
            norm_stats_file_path, 
            action_dim=7, 
            time_horizon=10,
            device='cuda'):
        super().__init__()
        with open(norm_stats_file_path, "r") as f:
            norm_stats = json.load(f)
        self.action_high, self.action_low = np.array(norm_stats["norm_stats"]["actions"]["q99"]), np.array(norm_stats["norm_stats"]["actions"]["q01"])
        self.action_high = self.action_high[:action_dim]
        self.action_low  = self.action_low[:action_dim]

        self.load_model(ckpt_path, action_dim=action_dim, time_horizon=time_horizon, device=device)
        self.error_action = np.zeros((time_horizon, action_dim))

    def load_model(self, ckpt_path = None, action_dim: int = 7, time_horizon: int = 10, device: str = 'cuda'):
        self.processor = AutoProcessor.from_pretrained("huiwon/ContextVLA-3B-Qwen2.5VL-FAST", use_fast=True)
        self.processor.tokenizer.padding_side = 'left'
        
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            "physical-intelligence/fast", trust_remote_code=True
        )
        self.fast_tokenizer.time_horizon = time_horizon
        self.fast_tokenizer.action_dim = action_dim

        self.model = modeling_contextvla.ContextVLA_Qwen2_5_VL.from_pretrained(
            "huiwon/ContextVLA-3B-Qwen2.5VL-FAST",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16
        ).to(device)

    def array_to_pil_image(self, frame):
        # frame shape: (C, H, W) -> (H, W, C)
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        return Image.fromarray(frame)

    def make_input(self, obs: dict) -> dict:
        main_pixel_values = obs["image_queue"]  # (8, 224, 224, 3)

        if "wrist_image_queue" in obs:
            wrist_pixel_values = obs["wrist_image_queue"]  # (8, 224, 224, 3)
        else:
            wrist_pixel_values = np.zeros_like(main_pixel_values)

        if "right_image_queue" in obs:
            right_pixel_values = obs["right_image_queue"]
        else:
            right_pixel_values = np.zeros_like(main_pixel_values)

        task_description = obs["task_description"]
        
        pixel_values = np.stack([main_pixel_values, wrist_pixel_values, right_pixel_values], axis=1)
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])

        image_contents = [{"type": "image", "image": self.array_to_pil_image(frame)} for frame in pixel_values]

        messages = [
            {
                "role": "user",
                "content": image_contents + [
                    {"type": "text", "text": task_description},
                ],
            }
        ]

        # Apply chat template to get the text input for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision information (depends on your process_vision_info function)
        image_inputs, video_inputs = vision_process.process_vision_info(messages)

        # Prepare inputs for the model using the main processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs
                    
    def extract_and_decode_action(self, generated_ids):
        action_indices = (ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= ACTION_TOKEN_MAX)
        action_indices = torch.where(action_indices)[0]
        
        output_action = self.fast_tokenizer.decode([generated_ids[0][action_indices] - ACTION_TOKEN_MIN])[0]

        if np.allclose(self.error_action, output_action):
            unnorm_actions = output_action
        else:
            unnorm_actions = (
                0.5 * (output_action + 1) * (self.action_high - self.action_low)
                + self.action_low
            )
        return np.array(unnorm_actions)
        
    def infer(self, obs: dict) -> dict:
        self.model.eval()

        inputs = self.make_input(obs)
        self.model.model.layers[2].input_id_context = inputs['input_ids'].detach()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        action = self.extract_and_decode_action(generated_ids)
        return action

    def reset(self) -> None:
        pass