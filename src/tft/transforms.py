import torch
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F
from typing import Union, Sequence, Optional, Dict, Any, List

class RandomCrop3D(Transform):
    """Crop a 3D volume tensor at a random location.

    The input tensor should have shape (..., C, D, H, W), where ... represents arbitrary leading
    batch dimensions, C is channels, and D, H, W are depth, height, and width. For example,
    a tensor with shape (1, 1, 64, 64, 64) can be cropped to (1, 1, 48, 48, 48).

    Args:
        size (int or sequence): Desired crop size (depth, height, width). If an int, a cubic crop
            is made (size, size, size). If a sequence, it must have 3 integers.
        padding (int or sequence, optional): Padding on each border. If an int, all six directions
            (left, right, top, bottom, front, back) are padded equally. If a sequence of length 3,
            it specifies (pad_width, pad_height, pad_depth) symmetrically. If length 6, it specifies
            (left, right, top, bottom, front, back). Default is None.
        pad_if_needed (bool, optional): Pad the input if smaller than the crop size. Default is False.
        fill (int, optional): Fill value for 'constant' padding mode. Default is 0.
        padding_mode (str, optional): Padding mode, either 'constant' or 'reflect'. Default is 'constant'.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        padding: Optional[Union[int, Sequence[int]]] = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant"
    ) -> None:
        super().__init__()
        
        # Set up crop size as (depth, height, width)
        self.size = self._setup_size_3d(size)
        
        # Parse padding into a tuple of 6 ints or None
        self.padding = self._parse_pad_padding_3d(padding) if padding is not None else None
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self._fill = fill  # Simplified; assumes fill is a scalar for tensors
        self.padding_mode = padding_mode
        
        # Validate padding mode
        if padding_mode not in ["constant", "reflect"]:
            raise ValueError("Padding mode must be 'constant' or 'reflect'.")

    def _setup_size_3d(self, size: Union[int, Sequence[int]]) -> tuple[int, int, int]:
        """Convert size to a tuple of 3 integers."""
        if isinstance(size, int):
            return (size, size, size)
        elif isinstance(size, (list, tuple)) and len(size) == 3:
            return tuple(size)
        else:
            raise ValueError("Size must be an int or a sequence of 3 ints.")

    def _parse_pad_padding_3d(self, padding: Union[int, Sequence[int]]) -> tuple[int, ...]:
        """Parse padding into a tuple of 6 ints: (left, right, top, bottom, front, back)."""
        if isinstance(padding, int):
            return (padding,) * 6
        elif isinstance(padding, (list, tuple)):
            if len(padding) == 3:
                pad_w, pad_h, pad_d = padding
                return (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d)
            elif len(padding) == 6:
                return tuple(padding)
            else:
                raise ValueError("Padding sequence must have length 3 or 6.")
        else:
            raise ValueError("Padding must be an int or a sequence.")

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        """Generate parameters for padding and cropping."""
        # Get original spatial dimensions
        original_D, original_H, original_W = self._query_size_3d(flat_inputs)
        crop_D, crop_H, crop_W = self.size

        # Initialize padding values
        if self.padding is not None:
            pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = self.padding
        else:
            pad_left = pad_right = pad_top = pad_bottom = pad_front = pad_back = 0

        # Compute padded sizes
        padded_D = original_D + pad_front + pad_back
        padded_H = original_H + pad_top + pad_bottom
        padded_W = original_W + pad_left + pad_right

        # Pad if needed to ensure crop size is achievable
        if self.pad_if_needed:
            if padded_D < crop_D:
                diff = crop_D - padded_D
                pad_front += diff
                pad_back += diff
                padded_D += 2 * diff
            if padded_H < crop_H:
                diff = crop_H - padded_H
                pad_top += diff
                pad_bottom += diff
                padded_H += 2 * diff
            if padded_W < crop_W:
                diff = crop_W - padded_W
                pad_left += diff
                pad_right += diff
                padded_W += 2 * diff

        # Check if crop is possible
        if padded_D < crop_D or padded_H < crop_H or padded_W < crop_W:
            raise ValueError(f"Crop size {self.size} is larger than padded size {(padded_D, padded_H, padded_W)}.")

        # Determine random crop starting points
        needs_depth_crop, front = (
            (True, int(torch.randint(0, padded_D - crop_D + 1, size=())))
            if padded_D > crop_D else (False, 0)
        )
        needs_height_crop, top = (
            (True, int(torch.randint(0, padded_H - crop_H + 1, size=())))
            if padded_H > crop_H else (False, 0)
        )
        needs_width_crop, left = (
            (True, int(torch.randint(0, padded_W - crop_W + 1, size=())))
            if padded_W > crop_W else (False, 0)
        )

        needs_crop = needs_depth_crop or needs_height_crop or needs_width_crop
        padding = [pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back]
        needs_pad = any(p > 0 for p in padding)

        return {
            "needs_pad": needs_pad,
            "padding": padding,
            "needs_crop": needs_crop,
            "front": front,
            "top": top,
            "left": left,
            "depth": crop_D,
            "height": crop_H,
            "width": crop_W
        }

    def transform(self, inpt: torch.Tensor, params: Dict[str, Any]) -> torch.Tensor:
        """Apply padding and cropping to the input tensor."""
        if params["needs_pad"]:
            inpt = F.pad(
                inpt,
                padding=params["padding"],
                fill=self._fill,
                padding_mode=self.padding_mode
            )
        if params["needs_crop"]:
            front, top, left = params["front"], params["top"], params["left"]
            depth, height, width = params["depth"], params["height"], params["width"]
            inpt = inpt[..., front:front + depth, top:top + height, left:left + width]
        return inpt

    def _query_size_3d(self, flat_inputs: List[Any]) -> tuple[int, int, int]:
        """Extract spatial dimensions from the input tensor."""
        inpt = flat_inputs[0]
        if isinstance(inpt, torch.Tensor):
            return inpt.shape[-3:]  # Last 3 dimensions: D, H, W
        else:
            raise TypeError("Input must be a torch.Tensor.")