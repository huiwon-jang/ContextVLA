from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
import logging
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable, ClassVar

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools
from transformers import AutoProcessor

from src.shared import array_typing as at
from src.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

import torch
from PIL import Image


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


def map_fast_token_to_vlm_action(tokens) -> str:
    """Maps fast action tokens to the VLM action format.
    Action token 0 is mapped to the string <robot_action_0>  ... and so on 
    """
    return ''.join([f"<robot_action_{token}>" for token in tokens])


def strong_augment(image):
    height, width = image.shape[1:3]

    crop_height = int(height * 0.95)
    crop_width = int(width * 0.95)

    max_h = height - crop_height
    max_w = width - crop_width
    if max_h > 0 and max_w > 0:
        start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
        start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
        image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

    image = torch.nn.functional.interpolate(
        image.permute(0, 3, 1, 2),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    
    angle = torch.rand(1, device=image.device) * 10 - 5 
    if torch.abs(angle) > 0.1:
        angle_rad = angle * torch.pi / 180.0

        # Create rotation matrix
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Apply rotation using grid_sample
        grid_x = torch.linspace(-1, 1, width, device=image.device)
        grid_y = torch.linspace(-1, 1, height, device=image.device)

        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

        # Expand to batch dimension
        grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

        # Apply rotation transformation
        grid_x_rot = grid_x * cos_a - grid_y * sin_a
        grid_y_rot = grid_x * sin_a + grid_y * cos_a

        # Stack and reshape for grid_sample
        grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

        image = torch.nn.functional.grid_sample(
            image.permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
    image = image * brightness_factor

    contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast_factor + mean

    saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
    gray = image.mean(dim=-1, keepdim=True)
    image = gray + (image - gray) * saturation_factor

    # Clamp values to [0, 1]
    image = torch.clamp(image, 0, 1)

    return image


def weak_augment(image):
    brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
    image = image * brightness_factor

    # Random contrast
    # Use tensor operations instead of .item() for torch.compile compatibility
    contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
    mean = image.mean(dim=[1, 2, 3], keepdim=True)
    image = (image - mean) * contrast_factor + mean

    saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
    gray = image.mean(dim=-1, keepdim=True)
    image = gray + (image - gray) * saturation_factor

    # Clamp values to [0, 1]
    image = torch.clamp(image, 0, 1)

    return image


def process_example(prompt, actions, image, fast_tokenizer):
    """Processes a single example from the dataset."""
    fast_tokens = fast_tokenizer(actions)
    #vlm_action = map_fast_token_to_vlm_action(fast_tokens[0])
    vlm_action = ''.join([map_fast_token_to_vlm_action(tokens) for tokens in fast_tokens])

    if 'base_0_rgb' in image: #libero, bridge
        main_pixel_values = np.stack([image[f"base_{i}_rgb"] for i in range(8)], axis=0) / 255.0
        main_pixel_values = torch.tensor(main_pixel_values, dtype=torch.float32)
        main_pixel_values = strong_augment(main_pixel_values)
        
        wrist_pixel_values = np.stack([image[f"left_wrist_{i}_rgb"] for i in range(8)], axis=0) / 255.0
        wrist_pixel_values = torch.tensor(wrist_pixel_values, dtype=torch.float32)
        wrist_pixel_values = weak_augment(wrist_pixel_values)
        
        right_pixel_values = np.stack([image[f"right_wrist_{i}_rgb"] for i in range(8)], axis=0) / 255.0
        right_pixel_values = torch.tensor(right_pixel_values, dtype=torch.float32)
        right_pixel_values = weak_augment(right_pixel_values)
    else: #robocasa
        main_pixel_values = np.stack([image[f"left_{i}_rgb"] for i in range(8)], axis=0) / 255.0
        main_pixel_values = torch.tensor(main_pixel_values, dtype=torch.float32)
        main_pixel_values = strong_augment(main_pixel_values)
        
        wrist_pixel_values = np.stack([image[f"right_{i}_rgb"] for i in range(8)], axis=0) / 255.0
        wrist_pixel_values = torch.tensor(wrist_pixel_values, dtype=torch.float32)
        wrist_pixel_values = strong_augment(wrist_pixel_values)
        
        right_pixel_values = np.stack([image[f"wrist_{i}_rgb"] for i in range(8)], axis=0) / 255.0
        right_pixel_values = torch.tensor(right_pixel_values, dtype=torch.float32)
        right_pixel_values = weak_augment(right_pixel_values)
    
    pixel_values = np.stack([main_pixel_values, wrist_pixel_values, right_pixel_values], axis=1)
    pixel_values = pixel_values.reshape(-1, pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
    
    pixel_values = (pixel_values * 255.0).astype(np.uint8)

    image_contents = [{"type": "image", "image": Image.fromarray(frame)} for frame in pixel_values]

    messages = [
        {
            "role": "user",
            "content": image_contents + [
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": vlm_action},
            ],
        },
    ]
    return messages


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    fast_tokenizer: ClassVar = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()
        
        self.fast_tokenizer.time_horizon = self.action_horizon
        self.fast_tokenizer.action_dim = self.action_dim

        message = process_example(prompt, data["actions"], data["image"], self.fast_tokenizer)

        return message


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    fast_tokenizer: ClassVar = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True
    )
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
