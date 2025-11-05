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
class BridgeInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    action_dim: int

    def __call__(self, data: dict) -> dict:

        if len(data["observation/image"].shape) == 3:
            data["observation/image"] = data["observation/image"][None,...]
        base_images = {
            f"base_{i}_rgb": _parse_image(image)
            for i, image in enumerate(data["observation/image"])
        }
        zero_image = np.zeros_like(next(iter(base_images.values())))

        wrist_images = {
            f"left_wrist_{i}_rgb": _parse_image(zero_image)
            for i, image in enumerate(data["observation/image"])
        }
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
class BridgeOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}
