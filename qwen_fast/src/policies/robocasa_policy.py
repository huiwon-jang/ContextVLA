import dataclasses

import einops
import numpy as np

from src import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        left_image = {
            f"left_{i}_rgb": _parse_image(image)
            for i, image in enumerate(data["video.left_view"])
        }
        right_image = {
            f"right_{i}_rgb": _parse_image(image)
            for i, image in enumerate(data["video.right_view"])
        }
        wrist_image = {
            f"wrist_{i}_rgb": _parse_image(image)
            for i, image in enumerate(data["video.wrist_view"])
        }

        inputs = {
            "image": {
                **left_image,
                **right_image,
                **wrist_image,
            },
        }
        
        if "annotation.human.action.task_description" in data:
            inputs["prompt"] = data["annotation.human.action.task_description"]
        
        if 'action' in data:
            action_base_motion = data["action"][:,:4]
            action_control_mode = data["action"][:,4:5] #binary
            action_end_eff_pos = data["action"][:,5:8]
            action_end_eff_rot = data["action"][:,8:11] #axis angle
            action_gripper_close = data["action"][:,11:12] #binary

            inputs["actions"] = np.concatenate([action_base_motion, action_control_mode, action_end_eff_pos, action_end_eff_rot, action_gripper_close], axis=1)
            inputs["actions"] = transforms.pad_to_dim(inputs["actions"], self.action_dim)
        
        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Robocasa, return 12 actions.
        return {"actions": np.asarray(data["actions"][..., :12])}
