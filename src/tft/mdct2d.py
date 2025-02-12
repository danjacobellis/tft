import math
import torch
import torch.nn.functional as F
from tft.mdct import _dct_2_unscaled

# -------------------------------------------------------------------
# 1) Re-use your 1D routines (dct_type_iv, etc.) as building blocks
# -------------------------------------------------------------------

def vorbis_window(window_length: int,
                  *,
                  dtype: torch.dtype = torch.float32,
                  device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Same as your 1D Vorbis window function.
    """
    n = torch.arange(window_length, dtype=dtype, device=device)
    N = float(window_length)
    sin_term = torch.sin(math.pi / N * (n + 0.5))
    return torch.sin((math.pi / 2.0) * sin_term.pow(2.0))

def dct_type_iv(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    """
    Same as your 1D DCT-IV along the last dimension.
    """
    # (omitting your internal _dct_2_unscaled for brevity,
    #  but assume it's included exactly as in your code.)
    N = x.shape[-1]
    # step 1: DCT-II with length=2*N
    dct2_full = _dct_2_unscaled(x, n=2*N)  # shape [..., 2*N]
    # step 2: slice odd indices => shape [..., N]
    dct4 = dct2_full[..., 1::2]
    # optional orthonormal
    if norm == "ortho":
        scale = math.sqrt(0.5) / math.sqrt(float(N))
        dct4 = dct4 * scale
    return dct4

def idct_type_iv(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    """
    Inverse of DCT-IV, which is (for real signals) the same as
    a forward DCT-IV up to scaling.  We can re-use dct_type_iv.
    """
    # For real signals, DCT-IV is its own inverse up to a factor:
    #   x = IDCT4(X)  ==>  x = DCT4(X) * (some scale).
    # Typically if norm=None, we have the factor 1/(2N).  If norm="ortho",
    # then it's its own inverse exactly.  For the sake of consistency with
    # your code, we replicate the same approach used in inverse_mdct:
    N = x.shape[-1]
    if norm is None:
        # do forward DCT-IV then scale
        out = dct_type_iv(x, norm=None)
        # your code used: out *= 0.5 / N  for the MDCT half-len case
        # but if x has size N, the factor is 0.5/N for each dimension.
        # We'll just replicate that logic. 
        out = out * (0.5 / float(N))
    elif norm == "ortho":
        # exactly invert for ortho
        out = dct_type_iv(x, norm='ortho')
    else:
        raise ValueError("norm must be None or 'ortho'.")
    return out

# -------------------------------------------------------------------
# 2) 2D framing and overlap-add
# -------------------------------------------------------------------

def frame2d(x: torch.Tensor,
            frame_height: int,
            frame_width: int,
            pad_end: bool = False) -> torch.Tensor:
    """
    2D framing with 50% overlap in each dimension.
    
    If `x` has shape [..., H, W], this returns
      [..., frames_h, frames_w, frame_height, frame_width].
    We use step = frame_height//2, frame_width//2.
    """
    if frame_height % 2 != 0 or frame_width % 2 != 0:
        raise ValueError("frame2d: frame_height, frame_width must be even "
                         "to do 50% overlap.")

    *batch_dims, H, W = x.shape
    step_h = frame_height // 2
    step_w = frame_width // 2

    # Optionally pad along bottom/right to fit an integer number of steps
    if pad_end:
        # vertical padding
        remainder_h = (H - frame_height) % step_h
        if remainder_h != 0:
            pad_h = step_h - remainder_h
        else:
            pad_h = 0
        # horizontal padding
        remainder_w = (W - frame_width) % step_w
        if remainder_w != 0:
            pad_w = step_w - remainder_w
        else:
            pad_w = 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad last two dims
            # shape now [..., H + pad_h, W + pad_w]
            H = x.shape[-2]
            W = x.shape[-1]

    # number of frames in each dimension
    frames_h = 1 + (H - frame_height)//step_h
    frames_w = 1 + (W - frame_width)//step_w

    # Collect frames in a python list
    blocks = []
    for row_idx in range(frames_h):
        row_start = row_idx * step_h
        row_end   = row_start + frame_height
        row_blocks = []
        for col_idx in range(frames_w):
            col_start = col_idx * step_w
            col_end   = col_start + frame_width
            patch = x[..., row_start:row_end, col_start:col_end]
            # shape [..., frame_height, frame_width]
            row_blocks.append(patch.unsqueeze(-3))  
            # unsqueeze(-3) so that we can cat along a new dimension for frames_w
        row_stack = torch.cat(row_blocks, dim=-3)  
        # shape [..., frames_w, frame_height, frame_width]
        blocks.append(row_stack.unsqueeze(-4))
        # unsqueeze(-4) so we can cat along a new dimension for frames_h

    out = torch.cat(blocks, dim=-4)
    # shape [..., frames_h, frames_w, frame_height, frame_width]
    return out

def overlap_and_add2d(frames_2d: torch.Tensor,
                      frame_height: int,
                      frame_width: int) -> torch.Tensor:
    """
    The 2D overlap-add inverse of `frame2d`.
    Input shape:  [..., frames_h, frames_w, frame_height, frame_width].
    Output shape: [..., (frames_h-1)*frame_height/2 + frame_height,
                       (frames_w-1)*frame_width/2 + frame_width ]
                  i.e. we used 50% overlap => step = frame_height/2, ...
    """
    if frame_height % 2 != 0 or frame_width % 2 != 0:
        raise ValueError("overlap_and_add2d expects even frame dims (for 50%).")

    *batch_dims, frames_h, frames_w, fh, fw = frames_2d.shape
    step_h = fh // 2
    step_w = fw // 2

    out_h = (frames_h - 1)*step_h + fh
    out_w = (frames_w - 1)*step_w + fw

    out = torch.zeros(*batch_dims, out_h, out_w,
                      dtype=frames_2d.dtype, device=frames_2d.device)

    for i in range(frames_h):
        row_start = i*step_h
        row_end   = row_start + fh
        for j in range(frames_w):
            col_start = j*step_w
            col_end   = col_start + fw
            out[..., row_start:row_end, col_start:col_end] += frames_2d[..., i, j, :, :]

    return out

# -------------------------------------------------------------------
# 3) 2D rearrangement (split into a,b,c,d for each dimension)
# -------------------------------------------------------------------

def _mdct_rearrange_1d(x: torch.Tensor) -> torch.Tensor:
    """
    The same 1D rearrangement done in `mdct`:
      Split x (last dim) into (a,b,c,d), each of size last_dim//4,
      then out = cat( [ -reverse(c) - d,  a -reverse(b) ], dim=-1 ).
    """
    L = x.shape[-1]
    if L % 4 != 0:
        raise ValueError("last dimension must be multiple of 4.")
    quarter = L // 4
    a, b, c, d = torch.split(x, quarter, dim=-1)
    rev_c = c.flip(dims=(-1,))
    rev_b = b.flip(dims=(-1,))
    out = torch.cat([-rev_c - d, a - rev_b], dim=-1)
    return out

def mdct_rearrange_2d(patches: torch.Tensor) -> torch.Tensor:
    """
    Apply the 1D rearrangement *first along width*, then along height.
    `patches` shape: [..., frame_height, frame_width].
    Return shape: [..., frame_height//2, frame_width//2].
    """
    # 1) rearrange along width (last dim)
    tmp = _mdct_rearrange_1d(patches)
    # now shape [..., frame_height, frame_width//2]
    # 2) rearrange along height. We want to treat the "frame_height" dim
    #    as the last dimension, so we can do a transpose, rearrange, transpose back.
    #    Alternatively, we can rearrange in a loop. But let's do the transpose approach.
    tmp_t = tmp.transpose(-2, -1)
    # shape [..., frame_width//2, frame_height]
    tmp_t2 = _mdct_rearrange_1d(tmp_t)
    # shape [..., frame_width//2, frame_height//2]
    # transpose back
    out = tmp_t2.transpose(-2, -1)
    # shape [..., frame_height//2, frame_width//2]
    return out

def _mdct_reverse_rearrange_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Reverse of `_mdct_rearrange_1d`.  If
       y = cat( [ -rev(c) - d, a - rev(b) ], dim=-1 ),
    we want to recover (a,b,c,d).  One can solve or simply
    replicate the logic from inverse_mdct.  For your
    inverse code, effectively each half is split into two:
      y0, y1 = torch.split(y, L/2, dim=-1)
      # each shape L/2
      => we want a, b, c, d of length L/4 each.
    This code must invert the signs and flips to get the original a,b,c,d.
    """
    # We'll adapt from your inverse_mdct logic for the 1D case:
    # in inverse_mdct, after iDCT-IV we had x0,x1 => real_frames = cat([x1, -rev(x1), -rev(x0), -x0], dim=-1).
    # But that was for half-len stuff. The easiest is to read off from the forward rearrangement:

    L2 = x.shape[-1]  # This is half the original length
    # so original length = 2*L2, quarters = L2/2
    quarter = L2 // 2
    y0, y1 = torch.split(x, quarter, dim=-1)
    # y0 = -rev(c) - d
    # y1 =  a - rev(b)
    # let's call them Y0, Y1, each shape [..., quarter].

    # We'll reconstruct a,b,c,d each shape "quarter/2".
    # quarter = (L // 2). So the original L was 2*L2 = 4*quarter.
    # Let half_quarter = quarter//2
    half_q = quarter // 2

    # Let c_part = rev(-y0[ : half_q ])... it's messy to code. 
    # For brevity: we know from the forward code:
    #   a = a
    #   b = b
    #   c = c
    #   d = d
    #   forward => out = [ -rev(c)-d, a-rev(b) ]
    #
    # We can decode as:
    #   -rev(c) - d = y0
    #   a - rev(b)  = y1
    # Then   -rev(c) = y0 + d
    #        c = rev(-y0 - d)
    # etc.

    # A simpler approach is to do exactly what the inverse_mdct code did for 1D after iDCT-IV,
    # i.e. reading that final "real_frames" line.  However, to keep this snippet clear,
    # we'll do a small system of equations.  A short route is to emulate the actual usage,
    # or do partial logic.  Because of the length, let's do a direct approach:

    raise NotImplementedError(
        "For a complete inverse, you need to invert the signs/flips from `_mdct_rearrange_1d`. "
        "See `inverse_mdct2d` below for how we do it in one shot after iDCT-IV."
    )


# -------------------------------------------------------------------
# 4) 2D DCT-IV and iDCT-IV (separable approach)
# -------------------------------------------------------------------

def dct_type_iv_2d(patches_2d: torch.Tensor,
                   norm: str = None) -> torch.Tensor:
    """
    Applies 1D DCT-IV along the last dimension, then along the second-last dimension.
    Input shape:  [..., H, W].
    Output shape: [..., H, W] (same shape).
    """
    # 1) dct_type_iv along width (last dim)
    out = dct_type_iv(patches_2d, norm=norm)
    # now shape [..., H, W]
    # 2) transpose and dct_type_iv along the last dim => effectively along height
    out_t = out.transpose(-2, -1)
    out_t2 = dct_type_iv(out_t, norm=norm)
    return out_t2.transpose(-2, -1)

def idct_type_iv_2d(patches_2d: torch.Tensor,
                    norm: str = None) -> torch.Tensor:
    """
    Inverse 2D DCT-IV (separable). If norm=None, applies the same factors
    used in `inverse_mdct` logic dimension by dimension.
    """
    # 1) iDCT-IV along width
    out = idct_type_iv(patches_2d, norm=norm)
    # 2) iDCT-IV along height
    out_t = out.transpose(-2, -1)
    out_t2 = idct_type_iv(out_t, norm=norm)
    return out_t2.transpose(-2, -1)

# -------------------------------------------------------------------
# 5) The actual 2D MDCT and inverse
# -------------------------------------------------------------------

def mdct2d(signals: torch.Tensor,
           frame_height: int,
           frame_width: int,
           window_fn = vorbis_window,
           pad_end: bool = False,
           norm: str = None) -> torch.Tensor:
    """
    2D MDCT using block-based approach (50% overlap in each dimension),
    extending your 1D `mdct` to 2D.

    Input shape:  [..., height, width]
    Output shape: [..., frames_h, frames_w, frame_height//2, frame_width//2]
    """
    # checks
    if (frame_height % 4 != 0) or (frame_width % 4 != 0):
        raise ValueError("2D MDCT requires frame_height and frame_width "
                         "to be multiples of 4.")

    # 1) 2D frame => shape [..., frames_h, frames_w, frame_height, frame_width]
    framed = frame2d(signals, frame_height, frame_width, pad_end=pad_end)

    # 2) Window (2D).  If `window_fn=None`, TF’s code used 1/sqrt(2), but
    #    we'll do that only if you specifically pass `None`.
    if window_fn is not None:
        wrow = window_fn(frame_height, dtype=framed.dtype, device=framed.device)
        wcol = window_fn(frame_width,  dtype=framed.dtype, device=framed.device)
        # shape (frame_height,) and (frame_width,)
        w2d = wrow.unsqueeze(-1) * wcol.unsqueeze(0)  # outer product => [H_f, W_f]
        framed = framed * w2d
    else:
        # fallback scale
        framed = framed * (1.0 / math.sqrt(2.0))

    # 3) Rearrange each patch dimension by dimension => shape [..., frames_h, frames_w, H_f//2, W_f//2]
    rearranged = mdct_rearrange_2d(framed)

    # 4) 2D DCT-IV on each patch => same shape
    #    We'll do it patchwise, but to keep it batched, we can just flatten
    #    the "frames_h, frames_w" as part of the leading dims:
    # shape = [..., frames_h, frames_w, (H_f/2), (W_f/2)] => treat the last 2 dims as the "2D" we transform.
    # We can do:
    shp = rearranged.shape
    *batch, fh, fw, h2, w2 = shp
    # flatten frames => shape [batch..., (fh*fw), h2, w2]
    rearranged_4d = rearranged.view(*batch, fh*fw, h2, w2)

    # apply 2D DCT-IV
    transformed_4d = dct_type_iv_2d(rearranged_4d, norm=norm)

    # reshape back
    out = transformed_4d.view(*batch, fh, fw, h2, w2)
    return out

def inverse_mdct2d(mdct_patches: torch.Tensor,
                   window_fn = vorbis_window,
                   norm: str = None) -> torch.Tensor:
    """
    Inverse of `mdct2d`.  Expects shape [..., frames_h, frames_w, H_f/2, W_f/2].
    Returns shape [..., out_height, out_width].
    """
    # figure out frame_height, frame_width from the last 2 dims
    *batch, fh, fw, h2, w2 = mdct_patches.shape
    frame_height = 2*h2
    frame_width  = 2*w2
    if (frame_height % 2 != 0) or (frame_width % 2 != 0):
        raise ValueError("inverse_mdct2d: half-len must be even => original frames must be multiple of 4.")

    # 1) iDCT-IV 2D (separable) => shape still [..., frames_h, frames_w, H_f/2, W_f/2]
    # flatten frames for batch
    patches_4d = mdct_patches.view(*batch, fh*fw, h2, w2)
    time_domain_4d = idct_type_iv_2d(patches_4d, norm=norm)
    # shape [..., fh*fw, H_f/2, W_f/2]
    time_domain = time_domain_4d.view(*batch, fh, fw, h2, w2)

    # 2) Reverse rearrangement + sign flipping + flipping, etc.
    #    In 1D, your code simply does something like:
    #        real_frames = cat([ x1, -rev(x1), -rev(x0), -x0 ], dim=-1)
    #    but that's after it splits in two chunks.  In 2D, we do dimension-by-dimension.
    #
    #    We'll do the *same pattern* that the forward code does, but in reverse:
    #       (a) separate the last dims into two parts => (x0, x1)
    #       (b) reassemble them into [H_f, W_f] with the appropriate signs.
    #
    #    Because it's fairly involved, we can mimic exactly what the 1D code’s
    #    `inverse_mdct` does: it merges the "halves," flips, and applies signs.
    #
    #    We'll do so in a helper function for clarity:

    real_frames = _inverse_mdct2d_reassemble(time_domain)  
    # shape [..., frames_h, frames_w, H_f, W_f]

    # 3) Window again
    if window_fn is not None:
        wrow = window_fn(frame_height, dtype=real_frames.dtype, device=real_frames.device)
        wcol = window_fn(frame_width,  dtype=real_frames.dtype, device=real_frames.device)
        w2d = wrow.unsqueeze(-1) * wcol.unsqueeze(0)
        real_frames = real_frames * w2d
    else:
        real_frames = real_frames * (1.0 / math.sqrt(2.0))

    # 4) Overlap-add
    signal = overlap_and_add2d(real_frames, frame_height, frame_width)
    return signal

def _inverse_mdct2d_reassemble(time_domain: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the 2D rearrangement from `mdct2d`, done dimension-by-dimension.
    `time_domain` shape: [..., frames_h, frames_w, H_f//2, W_f//2].

    We effectively replicate the logic from the 1D `inverse_mdct`:
      real_frames = cat([x1, -rev(x1), -rev(x0), -x0], dim=-1)
    but do it in 2D by:
      1) do the "width" dimension reassembly,
      2) do the "height" dimension reassembly.
    """
    # Step 1) Reassemble width
    # We'll treat W_f//2 as the last dimension, so we can do the same approach as 1D:
    #   split into x0,x1 each of size (W_f//4).
    # But we must ensure that W_f//2 is even => W_f % 4 == 0 by construction.
    x = time_domain
    *b, fh, fw, h2, w2 = x.shape
    # w2 = W_f//2
    # split w2 in half
    if w2 % 2 != 0:
        raise ValueError("Need w2 to be even => original frame_width multiple of 4.")
    half_w2 = w2 // 2
    # x0, x1 => shape [..., frames_h, frames_w, H_f//2, half_w2]
    x0 = x[..., :half_w2]
    x1 = x[..., half_w2:]
    # replicate your 1D pattern: real_frames_w = cat([ x1, -rev(x1), -rev(x0), -x0 ], dim=-1)
    # but we do this flipping along the last dimension:
    rev_x1 = x1.flip(dims=(-1,))
    rev_x0 = x0.flip(dims=(-1,))
    real_frames_w = torch.cat([x1, -rev_x1, -rev_x0, -x0], dim=-1)
    # shape [..., frames_h, frames_w, H_f//2, W_f]

    # Step 2) Reassemble height, but now the second-last dimension is H_f//2,
    # so we do the same pattern in that dimension. Let's transpose to make it last:
    real_frames_w_t = real_frames_w.transpose(-2, -1)
    # shape [..., W_f, H_f//2]

    # repeat the split:
    # now the last dimension is H_f//2, so we split it in half:
    h2_size = real_frames_w_t.shape[-1]
    if h2_size % 2 != 0:
        raise ValueError("Need H_f//2 to be even => original frame_height multiple of 4.")
    half_h2 = h2_size // 2
    x0h = real_frames_w_t[..., :half_h2]
    x1h = real_frames_w_t[..., half_h2:]
    rev_x1h = x1h.flip(dims=(-1,))
    rev_x0h = x0h.flip(dims=(-1,))

    real_frames_hw_t = torch.cat([x1h, -rev_x1h, -rev_x0h, -x0h], dim=-1)
    # shape [..., W_f, H_f]

    # transpose back
    real_frames_hw = real_frames_hw_t.transpose(-2, -1)
    # shape [..., H_f, W_f]

    return real_frames_hw


#
# End of code
#

