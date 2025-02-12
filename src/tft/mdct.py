import math
import torch
import torch.nn.functional as F

def _dct_2_unscaled(x: torch.Tensor, n: int) -> torch.Tensor:
    length_in = x.shape[-1]
    if length_in < n:
        pad_amount = n - length_in
        x = F.pad(x, (0, pad_amount))
    elif length_in > n:
        x = x[..., :n]
    X = torch.fft.rfft(x, n=2*n, dim=-1)
    X = X[..., :n]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    scale = 2.0 * torch.exp(-1j * (math.pi / (2.0 * n)) * k)
    return torch.real(X * scale)

def vorbis_window(window_length: int, *, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    n = torch.arange(window_length, dtype=dtype, device=device)
    N = float(window_length)
    sin_term = torch.sin(math.pi / N * (n + 0.5))
    return torch.sin((math.pi / 2.0) * sin_term.pow(2.0))

def frame(x: torch.Tensor, frame_length: int, frame_step: int, pad_end: bool = False) -> torch.Tensor:
    *batch_dims, num_samples = x.shape
    if pad_end:
        remainder = (num_samples - frame_length) % frame_step
        if remainder != 0:
            pad_size = frame_step - remainder
            x = F.pad(x, (0, pad_size))
            num_samples = x.shape[-1]
    num_frames = 1 + (num_samples - frame_length) // frame_step
    frames = []
    start = 0
    for _ in range(num_frames):
        end = start + frame_length
        frames.append(x[..., start:end].unsqueeze(-2))
        start += frame_step
    return torch.cat(frames, dim=-2)

def overlap_and_add(frames: torch.Tensor, frame_step: int) -> torch.Tensor:
    *batch_dims, num_frames, frame_length = frames.shape
    total_samples = (num_frames - 1) * frame_step + frame_length
    output = torch.zeros(*batch_dims, total_samples, dtype=frames.dtype, device=frames.device)
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        output[..., start:end] += frames[..., i, :]
    return output

def dct_type_iv(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    N = x.shape[-1]
    dct2 = _dct_2_unscaled(x, n=2 * N)
    dct4 = dct2[..., 1::2]
    if norm == "ortho":
        scale = math.sqrt(0.5) / math.sqrt(float(N))
        dct4 = dct4 * scale
    return dct4

def mdct(signals: torch.Tensor, frame_length: int, window_fn=vorbis_window, pad_end: bool = False, norm: str = None) -> torch.Tensor:
    if frame_length % 4 != 0:
        raise ValueError("frame_length must be multiple of 4 for this MDCT.")
    frame_step = frame_length // 2
    framed = frame(signals, frame_length, frame_step, pad_end=pad_end)
    if window_fn is not None:
        w = window_fn(frame_length, dtype=framed.dtype, device=framed.device)
        framed = framed * w
    else:
        framed = framed * (1.0 / math.sqrt(2.0))
    quarter_len = frame_length // 4
    a, b, c, d = torch.split(framed, quarter_len, dim=-1)
    def revlast(t):
        return t.flip(dims=(-1,))
    first_half = -revlast(c) - d
    second_half = a - revlast(b)
    frames_rearranged = torch.cat([first_half, second_half], dim=-1)
    mdct_out = dct_type_iv(frames_rearranged, norm=norm)
    return mdct_out

def inverse_mdct(mdcts: torch.Tensor, window_fn=vorbis_window, norm: str = None) -> torch.Tensor:
    half_len = mdcts.shape[-1]
    frame_length = 2 * half_len
    if norm is None:
        out = dct_type_iv(mdcts, norm=None)
        out = out * (0.5 / float(half_len))
    elif norm == "ortho":
        out = dct_type_iv(mdcts, norm="ortho")
    else:
        raise ValueError("norm must be None or 'ortho'.")
    if half_len % 2 != 0:
        raise ValueError("half_len must be even for this rearrangement, but got half_len=%d" % half_len)
    split_size = half_len // 2
    x0, x1 = torch.split(out, split_size, dim=-1)
    def revlast(t):
        return t.flip(dims=(-1,))
    real_frames = torch.cat([x1, -revlast(x1), -revlast(x0), -x0], dim=-1)
    if window_fn is not None:
        w = window_fn(frame_length, dtype=real_frames.dtype, device=real_frames.device)
        real_frames = real_frames * w
    else:
        real_frames = real_frames * (1.0 / math.sqrt(2.0))
    signal = overlap_and_add(real_frames, half_len)
    return signal

def idct_type_iv(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    N = x.shape[-1]
    if norm is None:
        out = dct_type_iv(x, norm=None)
        out = out * (0.5 / float(N))
    elif norm == "ortho":
        out = dct_type_iv(x, norm="ortho")
    else:
        raise ValueError("norm must be None or 'ortho'.")
    return out

def frame2d(x: torch.Tensor, frame_height: int, frame_width: int, pad_end: bool = False) -> torch.Tensor:
    if frame_height % 2 != 0 or frame_width % 2 != 0:
        raise ValueError("frame2d: frame_height, frame_width must be even to do 50% overlap.")
    *batch_dims, H, W = x.shape
    step_h = frame_height // 2
    step_w = frame_width // 2
    if pad_end:
        remainder_h = (H - frame_height) % step_h
        pad_h = step_h - remainder_h if remainder_h != 0 else 0
        remainder_w = (W - frame_width) % step_w
        pad_w = step_w - remainder_w if remainder_w != 0 else 0
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H = x.shape[-2]
            W = x.shape[-1]
    frames_h = 1 + (H - frame_height) // step_h
    frames_w = 1 + (W - frame_width) // step_w
    blocks = []
    for row_idx in range(frames_h):
        row_start = row_idx * step_h
        row_end = row_start + frame_height
        row_blocks = []
        for col_idx in range(frames_w):
            col_start = col_idx * step_w
            col_end = col_start + frame_width
            patch = x[..., row_start:row_end, col_start:col_end]
            row_blocks.append(patch.unsqueeze(-3))
        row_stack = torch.cat(row_blocks, dim=-3)
        blocks.append(row_stack.unsqueeze(-4))
    out = torch.cat(blocks, dim=-4)
    return out

def overlap_and_add2d(frames_2d: torch.Tensor, frame_height: int, frame_width: int) -> torch.Tensor:
    if frame_height % 2 != 0 or frame_width % 2 != 0:
        raise ValueError("overlap_and_add2d expects even frame dims (for 50%).")
    *batch_dims, frames_h, frames_w, fh, fw = frames_2d.shape
    step_h = fh // 2
    step_w = fw // 2
    out_h = (frames_h - 1) * step_h + fh
    out_w = (frames_w - 1) * step_w + fw
    out = torch.zeros(*batch_dims, out_h, out_w, dtype=frames_2d.dtype, device=frames_2d.device)
    for i in range(frames_h):
        row_start = i * step_h
        row_end = row_start + fh
        for j in range(frames_w):
            col_start = j * step_w
            col_end = col_start + fw
            out[..., row_start:row_end, col_start:col_end] += frames_2d[..., i, j, :, :]
    return out

def _mdct_rearrange_1d(x: torch.Tensor) -> torch.Tensor:
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
    tmp = _mdct_rearrange_1d(patches)
    tmp_t = tmp.transpose(-2, -1)
    tmp_t2 = _mdct_rearrange_1d(tmp_t)
    out = tmp_t2.transpose(-2, -1)
    return out

def _mdct_reverse_rearrange_1d(x: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("For a complete inverse, you need to invert the signs/flips from _mdct_rearrange_1d. See inverse_mdct2d below for how we do it in one shot after iDCT-IV.")

def dct_type_iv_2d(patches_2d: torch.Tensor, norm: str = None) -> torch.Tensor:
    out = dct_type_iv(patches_2d, norm=norm)
    out_t = out.transpose(-2, -1)
    out_t2 = dct_type_iv(out_t, norm=norm)
    return out_t2.transpose(-2, -1)

def idct_type_iv_2d(patches_2d: torch.Tensor, norm: str = None) -> torch.Tensor:
    out = idct_type_iv(patches_2d, norm=norm)
    out_t = out.transpose(-2, -1)
    out_t2 = idct_type_iv(out_t, norm=norm)
    return out_t2.transpose(-2, -1)

def mdct2d(signals: torch.Tensor, frame_height: int, frame_width: int, window_fn=vorbis_window, pad_end: bool = False, norm: str = None) -> torch.Tensor:
    if (frame_height % 4 != 0) or (frame_width % 4 != 0):
        raise ValueError("2D MDCT requires frame_height and frame_width to be multiples of 4.")
    framed = frame2d(signals, frame_height, frame_width, pad_end=pad_end)
    if window_fn is not None:
        wrow = window_fn(frame_height, dtype=framed.dtype, device=framed.device)
        wcol = window_fn(frame_width, dtype=framed.dtype, device=framed.device)
        w2d = wrow.unsqueeze(-1) * wcol.unsqueeze(0)
        framed = framed * w2d
    else:
        framed = framed * (1.0 / math.sqrt(2.0))
    rearranged = mdct_rearrange_2d(framed)
    shp = rearranged.shape
    *batch, fh, fw, h2, w2 = shp
    rearranged_4d = rearranged.view(*batch, fh * fw, h2, w2)
    transformed_4d = dct_type_iv_2d(rearranged_4d, norm=norm)
    out = transformed_4d.view(*batch, fh, fw, h2, w2)
    return out

def _inverse_mdct2d_reassemble(time_domain: torch.Tensor) -> torch.Tensor:
    x = time_domain
    *b, fh, fw, h2, w2 = x.shape
    if w2 % 2 != 0:
        raise ValueError("Need w2 to be even => original frame_width multiple of 4.")
    half_w2 = w2 // 2
    x0 = x[..., :half_w2]
    x1 = x[..., half_w2:]
    rev_x1 = x1.flip(dims=(-1,))
    rev_x0 = x0.flip(dims=(-1,))
    real_frames_w = torch.cat([x1, -rev_x1, -rev_x0, -x0], dim=-1)
    real_frames_w_t = real_frames_w.transpose(-2, -1)
    h2_size = real_frames_w_t.shape[-1]
    if h2_size % 2 != 0:
        raise ValueError("Need H_f//2 to be even => original frame_height multiple of 4.")
    half_h2 = h2_size // 2
    x0h = real_frames_w_t[..., :half_h2]
    x1h = real_frames_w_t[..., half_h2:]
    rev_x1h = x1h.flip(dims=(-1,))
    rev_x0h = x0h.flip(dims=(-1,))
    real_frames_hw_t = torch.cat([x1h, -rev_x1h, -rev_x0h, -x0h], dim=-1)
    real_frames_hw = real_frames_hw_t.transpose(-2, -1)
    return real_frames_hw

def inverse_mdct2d(mdct_patches: torch.Tensor, window_fn=vorbis_window, norm: str = None) -> torch.Tensor:
    *batch, fh, fw, h2, w2 = mdct_patches.shape
    frame_height = 2 * h2
    frame_width = 2 * w2
    if (frame_height % 2 != 0) or (frame_width % 2 != 0):
        raise ValueError("inverse_mdct2d: half-len must be even => original frames must be multiple of 4.")
    patches_4d = mdct_patches.view(*batch, fh * fw, h2, w2)
    time_domain_4d = idct_type_iv_2d(patches_4d, norm=norm)
    time_domain = time_domain_4d.view(*batch, fh, fw, h2, w2)
    real_frames = _inverse_mdct2d_reassemble(time_domain)
    if window_fn is not None:
        wrow = window_fn(frame_height, dtype=real_frames.dtype, device=real_frames.device)
        wcol = window_fn(frame_width, dtype=real_frames.dtype, device=real_frames.device)
        w2d = wrow.unsqueeze(-1) * wcol.unsqueeze(0)
        real_frames = real_frames * w2d
    else:
        real_frames = real_frames * (1.0 / math.sqrt(2.0))
    signal = overlap_and_add2d(real_frames, frame_height, frame_width)
    return signal
