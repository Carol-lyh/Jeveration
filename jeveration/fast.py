"""Speed-ups for inference (JeverationPipeline(fast=True)). Same model, small numerical differences (bfloat16 rounding):

    lora_to               keep the LoRA adapters in bfloat16, as autocast computes them anyway: identical results
    patch_linear_attention  without flash-linear-attention (e.g. Windows / ROCm, where Triton is missing), transformers
                          runs the linear-attention layers with a Python loop of ~250 tiny kernels per layer; here the loop
                          is a few batched matmuls
    store_autocast_weights  image head and decoders keep bfloat16 weights (they run under autocast only): identical results
    replace_slow_convs    the small convolutions without MIOpen, which compiles kernels for seconds on first use
    enable_gemm_tuning    ROCm: PyTorch TunableOp picks the fastest GEMM kernel per shape, kept in tuning/*.csv

The AMD flash / memory-efficient attention (AOTriton) is enabled in jeveration/__init__.py, before torch is imported.
"""
import torch
import torch.nn.functional as F

from .backbone import LoRALinear


@torch.no_grad()
def lora_to(lm, dtype=torch.bfloat16):
    """Store every LoRA adapter in `dtype`. The backbone runs under autocast, which casts the float32 adapters (and their
    input) to bfloat16 in every call anyway: same results, less memory traffic. (Merging the adapters into the bfloat16
    weights instead is NOT equivalent: their small updates drown in the rounding of the large weights.) -> count"""
    n = 0
    for m in lm.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.data, m.lora_B.data = m.lora_A.data.to(dtype), m.lora_B.data.to(dtype)
            n += 1
    return n


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def unit_lower_inverse(lower):
    """(I + L)^-1 for strictly lower triangular L [..., n, n], n a power of two, by block doubling: with the inverses A', B'
    of two neighbouring diagonal blocks known, their joint block's inverse has the lower-left block -B' C A'. log2(n) levels
    of batched matmuls, as stable as forward substitution. (A power series sum (-L)^k loses the result to cancellation;
    torch.linalg.solve_triangular is ~30x slower on ROCm.)"""
    n = lower.shape[-1]
    assert n & (n - 1) == 0, n
    m = lower + torch.eye(n, dtype=lower.dtype, device=lower.device)
    x = torch.eye(n, dtype=lower.dtype, device=lower.device).expand_as(lower).clone()
    lead = lower.shape[:-2]
    s = 1
    while s < n:
        nb = n // (2 * s)
        md = torch.diagonal(m.view(*lead, nb, 2 * s, nb, 2 * s), dim1=-4, dim2=-2)       # [..., 2s, 2s, nb], views
        xd = torch.diagonal(x.view(*lead, nb, 2 * s, nb, 2 * s), dim1=-4, dim2=-2)
        a_inv = xd[..., :s, :s, :].movedim(-1, -3)
        b_inv = xd[..., s:, s:, :].movedim(-1, -3)
        c = md[..., s:, :s, :].movedim(-1, -3)
        xd[..., s:, :s, :] = (-(b_inv @ c @ a_inv)).movedim(-3, -1)
        s *= 2
    return x


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False, **kwargs):
    """transformers' torch_chunk_gated_delta_rule (Qwen3.5) with the within-chunk forward substitution replaced by
    batched matmuls (unit_lower_inverse). Shapes as there: q, k [B, L, H, K], v [B, L, H, V], g, beta [B, L, H]."""
    dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)       # in the input dtype, as transformers does
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().float() for x in (query, key, value, beta, g)]
    b, h, n, dk = key.shape
    dv = value.shape[-1]
    pad = (chunk_size - n % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, g = F.pad(beta, (0, pad)), F.pad(g, (0, pad))
    nc = (n + pad) // chunk_size
    query = query * dk ** -0.5
    k_beta, v_beta = key * beta[..., None], value * beta[..., None]
    query, key, k_beta, v_beta = (x.reshape(b, h, nc, chunk_size, -1) for x in (query, key, k_beta, v_beta))
    g = g.reshape(b, h, nc, chunk_size).cumsum(-1)
    decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
    # (I + L) T = I with L the strictly lower part of (k_beta k^T * decay): solve for value and k_cumdecay directly
    lower = ((k_beta @ key.transpose(-1, -2)) * decay).tril(-1)
    rhs = torch.cat([v_beta, k_beta * g.exp()[..., None]], -1)
    sol = unit_lower_inverse(lower) @ rhs
    value, k_cumdecay = sol[..., :dv], sol[..., dv:]

    state = (torch.zeros(b, h, dk, dv, dtype=value.dtype, device=value.device) if initial_state is None
             else initial_state.to(value))
    out = torch.empty_like(value)
    causal = decay.tril()          # decay is already lower triangular, incl. the diagonal
    for i in range(nc):
        q_i, k_i, g_i = query[:, :, i], key[:, :, i], g[:, :, i]
        v_new = value[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = (q_i * g_i.exp()[..., None]) @ state + (q_i @ k_i.transpose(-1, -2) * causal[:, :, i]) @ v_new
        state = state * g_i[..., -1, None, None].exp() + (k_i * (g_i[..., -1, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
    out = out.reshape(b, h, -1, dv)[:, :, :n].transpose(1, 2).contiguous().to(dtype)
    return out, (state if output_final_state else None)


def patch_linear_attention(lm):
    """use chunk_gated_delta_rule in every linear-attention layer that runs transformers' torch fallback -> how many"""
    n = 0
    for mod in lm.modules():
        f = getattr(mod, "chunk_gated_delta_rule", None)
        if f is not None and getattr(f, "__name__", "") == "torch_chunk_gated_delta_rule":
            mod.chunk_gated_delta_rule = chunk_gated_delta_rule
            n += 1
    return n


@torch.no_grad()
def store_autocast_weights(module, dtype=torch.bfloat16):
    """Keep the weights of the linear / conv / attention layers of a module that only ever runs under autocast in the dtype
    autocast casts them to anyway: the same results, without re-casting ~0.5 GB of float32 weights in every call. -> count"""
    n = 0
    for m in module.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            params = [m.weight, m.bias]
        elif isinstance(m, torch.nn.MultiheadAttention):
            params = [m.in_proj_weight, m.in_proj_bias]
        else:
            continue
        for p in params:
            if p is not None and p.is_floating_point():
                p.data = p.data.to(dtype)
        n += 1
    return n


def enable_gemm_tuning(device, folder=None):
    """ROCm: PyTorch TunableOp times the rocBLAS / hipBLASLt candidates of every GEMM shape once and keeps the fastest in
    <folder>/tunableop_<arch><device index>.csv, read again by later processes (~25% less time per image on a Radeon 8060S).
    A shape not seen before costs a few seconds once. No-op on CUDA. -> the file or None"""
    import os
    if torch.version.hip is None or not hasattr(torch.cuda, "tunable"):
        return None
    folder = folder or os.environ.get("JEVERATION_TUNING_DIR") or os.path.join(os.path.dirname(os.path.dirname(__file__)), "tuning")
    os.makedirs(folder, exist_ok=True)
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    t = torch.cuda.tunable
    t.set_filename(os.path.join(folder, f"tunableop_{arch}.csv"), insert_device_ordinal=True)
    t.enable(True)
    t.tuning_enable(True)
    return t.get_filename()


class CausalDepthwiseConv1d(torch.nn.Module):
    """Drop-in for the linear-attention layers' nn.Conv1d(C, C, K, groups=C, padding=K-1), of whose output the caller keeps
    the first L positions: K shifted multiply-adds in float32. MIOpen compiles a kernel per sequence length for the
    original, seconds each on Windows."""

    def __init__(self, conv):
        super().__init__()
        assert conv.groups == conv.in_channels == conv.out_channels and conv.padding[0] == conv.kernel_size[0] - 1
        self.weight, self.bias, self.k = conv.weight, conv.bias, conv.kernel_size[0]

    def forward(self, x):                                   # [B, C, L] -> [B, C, L]
        n, w = x.shape[-1], self.weight[:, 0].float()     # [C, K]
        xp = F.pad(x.float(), (self.k - 1, 0))
        y = sum(w[:, j, None] * xp[..., j:j + n] for j in range(self.k))
        if self.bias is not None:
            y = y + self.bias.float()[:, None]
        return y.to(x.dtype)


class Im2colConv2d(torch.nn.Module):
    """Drop-in for a small nn.Conv2d (stride 1): im2col + one GEMM in float32, without MIOpen's kernel compilation (13 s per
    process for the decoders' final 3 -> 3 channel 3 x 3 convolution)."""

    def __init__(self, conv):
        super().__init__()
        assert conv.stride == (1, 1) and conv.dilation == (1, 1) and conv.groups == 1 and isinstance(conv.padding, tuple)
        self.weight, self.bias, self.kernel_size, self.padding = conv.weight, conv.bias, conv.kernel_size, conv.padding

    def forward(self, x):                                   # [B, Ci, H, W] -> [B, Co, H, W]
        b, _, h, w = x.shape
        cols = F.unfold(x.float(), self.kernel_size, padding=self.padding)         # [B, Ci*kh*kw, H*W]
        y = self.weight.float().flatten(1) @ cols
        if self.bias is not None:
            y = y + self.bias.float()[:, None]
        return y.view(b, -1, h, w)


def replace_slow_convs(module):
    """the linear-attention layers' conv1d and the decoders' small final conv2d -> the replacements above. -> count"""
    n = 0
    for m in list(module.modules()):
        for name, c in list(m.named_children()):
            if isinstance(c, torch.nn.Conv1d) and c.groups == c.in_channels > 1 and c.padding[0] == c.kernel_size[0] - 1:
                setattr(m, name, CausalDepthwiseConv1d(c))
            elif isinstance(c, torch.nn.Conv2d) and c.in_channels * c.out_channels <= 64 and c.kernel_size != (1, 1) \
                    and c.stride == (1, 1) and c.groups == 1:
                setattr(m, name, Im2colConv2d(c))
            else:
                continue
            n += 1
    return n
