import math
import torch
import torch.nn.functional as F
from einops import rearrange

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
    slices = []
    start = 0
    for _ in range(num_frames):
        end = start + frame_length
        slices.append(x[..., start:end])
        start += frame_step
    return rearrange(slices, 'f ... l -> ... f l')

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
    rearranged = rearrange(framed, '... (four q) -> ... four q', four=4, q=quarter_len)
    a = rearranged[..., 0, :]
    b = rearranged[..., 1, :]
    c = rearranged[..., 2, :]
    d = rearranged[..., 3, :]
    first_half = -c.flip(dims=(-1,)) - d
    second_half = a - b.flip(dims=(-1,))
    stacked = torch.stack([first_half, second_half], dim=-2)
    frames_rearranged = rearrange(stacked, '... h l -> ... (h l)')
    return dct_type_iv(frames_rearranged, norm=norm)

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
    split_size = half_len // 2
    splitted = rearrange(out, '... (two s) -> ... two s', two=2, s=split_size)
    x0 = splitted[..., 0, :]
    x1 = splitted[..., 1, :]
    real_frames_4 = [x1, -x1.flip(dims=(-1,)), -x0.flip(dims=(-1,)), -x0]
    stacked = torch.stack(real_frames_4, dim=-2)
    real_frames = rearrange(stacked, '... h l -> ... (h l)')
    if window_fn is not None:
        w = window_fn(frame_length, dtype=real_frames.dtype, device=real_frames.device)
        real_frames = real_frames * w
    else:
        real_frames = real_frames * (1.0 / math.sqrt(2.0))
    return overlap_and_add(real_frames, half_len)

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
    patches = []
    for row_idx in range(frames_h):
        row_start = row_idx * step_h
        row_end = row_start + frame_height
        for col_idx in range(frames_w):
            col_start = col_idx * step_w
            col_end = col_start + frame_width
            patch = x[..., row_start:row_end, col_start:col_end]
            patches.append(patch)
    return rearrange(patches, '(fh fw) ... h w -> ... fh fw h w', fh=frames_h, fw=frames_w)

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
    splitted = rearrange(x, '... (four q) -> ... four q', four=4, q=quarter)
    a = splitted[..., 0, :]
    b = splitted[..., 1, :]
    c = splitted[..., 2, :]
    d = splitted[..., 3, :]
    rev_c = c.flip(dims=(-1,))
    rev_b = b.flip(dims=(-1,))
    out_2 = [-rev_c - d, a - rev_b]
    stacked = torch.stack(out_2, dim=-2)
    return rearrange(stacked, '... h l -> ... (h l)')

def mdct_rearrange_2d(patches: torch.Tensor) -> torch.Tensor:
    tmp = _mdct_rearrange_1d(patches)
    tmp_t = rearrange(tmp, '... a b -> ... b a')
    tmp_t2 = _mdct_rearrange_1d(tmp_t)
    return rearrange(tmp_t2, '... a b -> ... b a')

def dct_type_iv_2d(patches_2d: torch.Tensor, norm: str = None) -> torch.Tensor:
    out = dct_type_iv(patches_2d, norm=norm)
    out_t = rearrange(out, '... a b -> ... b a')
    out_t2 = dct_type_iv(out_t, norm=norm)
    return rearrange(out_t2, '... a b -> ... b a')

def idct_type_iv_2d(patches_2d: torch.Tensor, norm: str = None) -> torch.Tensor:
    out = idct_type_iv(patches_2d, norm=norm)
    out_t = rearrange(out, '... a b -> ... b a')
    out_t2 = idct_type_iv(out_t, norm=norm)
    return rearrange(out_t2, '... a b -> ... b a')

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
    *batch, fh, fw, h2, w2 = rearranged.shape
    rearranged_4d = rearrange(rearranged, '... fh fw h2 w2 -> ... (fh fw) h2 w2')
    transformed_4d = dct_type_iv_2d(rearranged_4d, norm=norm)
    out = rearrange(transformed_4d, '... (fh fw) h2 w2 -> ... fh fw h2 w2', fh=fh, fw=fw)
    return out

def _inverse_mdct2d_reassemble(time_domain: torch.Tensor) -> torch.Tensor:
    x = time_domain
    *b, fh, fw, h2, w2 = x.shape
    half_w2 = w2 // 2
    splitted = rearrange(x, '... (two w2) -> ... two w2', two=2, w2=half_w2)
    x0 = splitted[..., 0, :]
    x1 = splitted[..., 1, :]
    real_frames_w_4 = [x1, -x1.flip(dims=(-1,)), -x0.flip(dims=(-1,)), -x0]
    stacked_w = rearrange(torch.stack(real_frames_w_4, dim=-2), '... h w -> ... (h w)')
    real_frames_w_t = rearrange(stacked_w, '... a b -> ... b a')
    h2_size = real_frames_w_t.shape[-1]
    half_h2 = h2_size // 2
    splitted_h = rearrange(real_frames_w_t, '... (two s) -> ... two s', two=2, s=half_h2)
    x0h = splitted_h[..., 0, :]
    x1h = splitted_h[..., 1, :]
    real_frames_hw_4 = [x1h, -x1h.flip(dims=(-1,)), -x0h.flip(dims=(-1,)), -x0h]
    stacked_h = rearrange(torch.stack(real_frames_hw_4, dim=-2), '... h w -> ... (h w)')
    real_frames_hw = rearrange(stacked_h, '... a b -> ... b a')
    return real_frames_hw

def inverse_mdct2d(mdct_patches: torch.Tensor, window_fn=vorbis_window, norm: str = None) -> torch.Tensor:
    *batch, fh, fw, h2, w2 = mdct_patches.shape
    frame_height = 2 * h2
    frame_width = 2 * w2
    if (frame_height % 2 != 0) or (frame_width % 2 != 0):
        raise ValueError("inverse_mdct2d: half-len must be even => original frames must be multiple of 4.")
    patches_4d = rearrange(mdct_patches, '... fh fw h2 w2 -> ... (fh fw) h2 w2')
    time_domain_4d = idct_type_iv_2d(patches_4d, norm=norm)
    time_domain = rearrange(time_domain_4d, '... (fh fw) h2 w2 -> ... fh fw h2 w2', fh=fh, fw=fw)
    real_frames = _inverse_mdct2d_reassemble(time_domain)
    if window_fn is not None:
        wrow = window_fn(frame_height, dtype=real_frames.dtype, device=real_frames.device)
        wcol = window_fn(frame_width, dtype=real_frames.dtype, device=real_frames.device)
        w2d = wrow.unsqueeze(-1) * wcol.unsqueeze(0)
        real_frames = real_frames * w2d
    else:
        real_frames = real_frames * (1.0 / math.sqrt(2.0))
    return overlap_and_add2d(real_frames, frame_height, frame_width)
