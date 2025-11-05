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
class LiberoInputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        base_images = {
            f"base_{i}_rgb": _parse_image(image)
            for i, image in enumerate(data["observation/image"])
        }
        wrist_images = {
            f"left_wrist_{i}_rgb": _parse_image(image)
            for i, image in enumerate(data["observation/wrist_image"])
        }
        zero_image = np.zeros_like(next(iter(base_images.values())))
        right_images = {
            f"right_wrist_{i}_rgb": _parse_image(zero_image)
            for i, image in enumerate(data["observation/image"])
        }

        inputs = {
            "image": {
                **base_images,
                **wrist_images,
                **right_images,
            },
        }
        
        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
