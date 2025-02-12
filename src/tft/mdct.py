import math
import torch
import torch.nn.functional as F

def vorbis_window(
    window_length: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    r"""
    Create the Vorbis power-complementary window of length `window_length`.

    Matches `tf.signal.vorbis_window`.  The formula:
      w[n] = sin\Bigl(\frac{\pi}{2}\,\sin^2\bigl(\tfrac{\pi}{N}(n+0.5)\bigr)\Bigr)
      for n = 0..N-1.
    """
    n = torch.arange(window_length, dtype=dtype, device=device)
    N = float(window_length)
    # sin_term = sin(pi/N * (n + 0.5))
    sin_term = torch.sin(math.pi / N * (n + 0.5))
    # window = sin( (pi/2) * (sin_term^2) )
    window = torch.sin((math.pi / 2.0) * sin_term.pow(2.0))
    return window

def frame(x: torch.Tensor,
          frame_length: int,
          frame_step: int,
          pad_end: bool = False) -> torch.Tensor:
    r"""
    Splits 1D or batched 1D signals into overlapping frames of length
    `frame_length` with hop `frame_step`. Operates on the last dimension.
    Returns a tensor of shape [..., num_frames, frame_length].
    """
    *batch_dims, num_samples = x.shape

    if pad_end:
        remainder = (num_samples - frame_length) % frame_step
        if remainder != 0:
            pad_size = frame_step - remainder
            # pad on the right along the last dimension
            x = F.pad(x, (0, pad_size))
            num_samples = x.shape[-1]

    num_frames = 1 + (num_samples - frame_length)//frame_step
    # Collect frames in a list (straightforward approach)
    frames = []
    start = 0
    for _ in range(num_frames):
        end = start + frame_length
        frames.append(x[..., start:end].unsqueeze(-2)) # shape [..., 1, frame_length]
        start += frame_step

    return torch.cat(frames, dim=-2)  # shape [..., num_frames, frame_length]

def overlap_and_add(frames: torch.Tensor,
                    frame_step: int) -> torch.Tensor:
    r"""
    Overlap‐add the frames along the last dimension to reconstruct
    the 1D signal. Inverse of `frame` (with the same step).
    Input shape: [..., num_frames, frame_length]
    Output shape: [..., samples]
    """
    *batch_dims, num_frames, frame_length = frames.shape
    total_samples = (num_frames - 1)*frame_step + frame_length

    output = torch.zeros(*batch_dims, total_samples,
                         dtype=frames.dtype, device=frames.device)
    for i in range(num_frames):
        start = i*frame_step
        end = start + frame_length
        output[..., start:end] += frames[..., i, :]
    return output

def _dct_2_unscaled(x: torch.Tensor, n: int) -> torch.Tensor:
    r"""
    Compute the "raw" DCT-II of `x` along the last dim, zero-padding/truncating
    to length `n`.  No orthonormal scaling is applied.

    If `x.shape[-1] < n`, we zero-pad; if `x.shape[-1] > n`, we truncate.
    Then:

      DCT-II(x)[k] = Re{ RFFT(x, length=2n)[k] * (2 * e^{-i pi/2n * k}) }
      (for k=0..n-1)

    """
    length_in = x.shape[-1]
    if length_in < n:
        # pad
        pad_amount = n - length_in
        x = F.pad(x, (0, pad_amount))
    elif length_in > n:
        # truncate
        x = x[..., :n]

    # 1) real FFT of length 2n
    X = torch.fft.rfft(x, n=2*n, dim=-1)  # shape [..., n+1]
    # keep only the first n bins
    X = X[..., :n]                        # shape [..., n]

    # 2) multiply by the scale factor
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    # scale = 2 * exp(-j * pi/(2n) * k)
    # (Comparing to TF, there's a small detail about "range(...) * pi * 0.5 / n")
    scale = 2.0 * torch.exp(-1j * (math.pi/(2.0*n)) * k)
    # 3) real part
    return torch.real(X * scale)

def dct_type_iv(x: torch.Tensor, norm: str = None) -> torch.Tensor:
    r"""
    Compute the DCT-IV of `x` along its last dimension, matching TF’s shape
    conventions. Input and output both have the same last-dim size = N.

    The math (unscaled) is:
      DCT4(x) := DCT2(x, length=2N)[..., 1::2]
      => shape is still [..., N].

    If norm=="ortho", multiply by sqrt(1/2) * (1/sqrt(N)).
    """
    N = x.shape[-1]
    # Step 1 & 2: DCT-II with length=2*N, then pick the odd indices
    dct2 = _dct_2_unscaled(x, n=2*N)  # shape [..., 2*N], but we slice to [:, :2*N]
    # Actually the returned shape is [..., 2*N], we only keep the first 2*N
    # (which is all of it). Next we do [..., 1::2].
    dct4 = dct2[..., 1::2]  # pick odd indices => shape [..., N]

    # Optional orthonormal scale
    if norm == "ortho":
        # multiply by sqrt(0.5) * 1/sqrt(N)
        scale = math.sqrt(0.5) / math.sqrt(float(N))
        dct4 = dct4 * scale

    return dct4

def mdct(signals: torch.Tensor,
         frame_length: int,
         window_fn = vorbis_window,
         pad_end: bool = False,
         norm: str = None) -> torch.Tensor:
    r"""
    MDCT ported from `tf.signal.mdct` for `frame_length % 4 == 0`.

    Returns shape [..., frames, frame_length//2].
    """
    if frame_length % 4 != 0:
        raise ValueError("frame_length must be multiple of 4 for this MDCT.")

    # 1) Frame the signal with 50% overlap
    frame_step = frame_length // 2
    framed = frame(signals, frame_length, frame_step, pad_end=pad_end)
    # shape => [..., frames, frame_length]

    # 2) Window
    if window_fn is not None:
        w = window_fn(frame_length, dtype=framed.dtype, device=framed.device)
        framed = framed * w
    else:
        # TF does 1/sqrt(2) if window_fn=None
        framed = framed * (1.0 / math.sqrt(2.0))

    # 3) Split into 4 equal parts (a,b,c,d), each size frame_length/4
    quarter_len = frame_length // 4
    a, b, c, d = torch.split(framed, quarter_len, dim=-1)

    # 4) Rearrange => shape [..., frames, frame_length//2]
    def revlast(t):
        return t.flip(dims=(-1,))
    first_half  = -revlast(c) - d   # shape [..., frames, quarter_len]
    second_half =  a - revlast(b)   # shape [..., frames, quarter_len]
    frames_rearranged = torch.cat([first_half, second_half], dim=-1)
    # => shape [..., frames, 2*quarter_len] = [..., frames, frame_length//2]

    # 5) Type‑IV DCT along last axis => returns same size [..., frames, frame_length//2]
    mdct_out = dct_type_iv(frames_rearranged, norm=norm)
    return mdct_out

def inverse_mdct(mdcts: torch.Tensor,
                 window_fn = vorbis_window,
                 norm: str = None) -> torch.Tensor:
    r"""
    Inverse of the above `mdct`, matching `tf.signal.inverse_mdct`.

    Input shape:  [..., frames, frame_length//2]
    Output shape: [..., samples], where samples = (frames-1)*(frame_length//2) + frame_length.

    Algorithm:
      1) half_len = mdcts.shape[-1], so frame_length = 2*half_len
      2) iDCT‑IV of mdcts.  In TF code:
         if norm=None:
           out = (0.5 / half_len) * dct_type_iv(mdcts, norm=None)
         else: # norm='ortho'
           out = dct_type_iv(mdcts, norm='ortho')
      3) Split that “out” into two chunks along last dim => each size half_len//2.
         Re-arrange => shape [..., frames, 2*half_len] = [..., frames, frame_length].
      4) Optional window
      5) overlap_and_add
    """
    half_len = mdcts.shape[-1]
    frame_length = 2*half_len

    # 1) iDCT-IV
    if norm is None:
        # TF applies 0.5/half_len afterwards
        out = dct_type_iv(mdcts, norm=None)
        out = out * (0.5 / float(half_len))
    elif norm == 'ortho':
        # No extra scale needed
        out = dct_type_iv(mdcts, norm='ortho')
    else:
        raise ValueError("norm must be None or 'ortho'.")

    # 2) Split into two equal parts along last axis
    if half_len % 2 != 0:
        raise ValueError("half_len must be even for this rearrangement, "
                         "but got half_len=%d" % half_len)
    split_size = half_len // 2
    x0, x1 = torch.split(out, split_size, dim=-1)
    # x0, x1 each => shape [..., frames, half_len//2]

    # 3) Concat => shape [..., frames, 2*half_len] = [..., frames, frame_length]
    def revlast(t):
        return t.flip(dims=(-1,))
    real_frames = torch.cat([
        x1,             # shape [..., split_size]
        -revlast(x1),
        -revlast(x0),
        -x0
    ], dim=-1)  # => shape [..., frames, 4*split_size = 2*half_len = frame_length]

    # 4) Window
    if window_fn is not None:
        w = window_fn(frame_length, dtype=real_frames.dtype, device=real_frames.device)
        real_frames = real_frames * w
    else:
        real_frames = real_frames * (1.0 / math.sqrt(2.0))

    # 5) Overlap-add with step = half_len
    signal = overlap_and_add(real_frames, half_len)
    return signal
