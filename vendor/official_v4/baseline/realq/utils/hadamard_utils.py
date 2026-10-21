import torch
import math
from pathlib import Path

try:
    import fast_hadamard_transform
except ImportError:
    fast_hadamard_transform = None


def _hadamard_transform_torch(u: torch.Tensor) -> torch.Tensor:
    x = u.contiguous()
    n = x.shape[-1]
    if not is_pow2(n):
        raise ValueError(f"Hadamard transform fallback expects power-of-2 last dim, got {n}")

    orig_shape = x.shape
    x = x.reshape(-1, n)
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        # Keep the fallback purely functional.  Besides avoiding mutation of
        # views, this is required when the transform is executed from a
        # custom autograd.Function's torch.func JVP rule.
        x = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
        h *= 2
    return x.view(orig_shape)


def _hadamard_transform(
    u: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    if fast_hadamard_transform is not None and u.is_cuda:
        # The CUDA extension applies scale while storing its output, avoiding
        # a separate normalization kernel and intermediate tensor traversal.
        return fast_hadamard_transform.hadamard_transform(
            u.contiguous(), scale=scale
        )
    output = _hadamard_transform_torch(u)
    return output if scale == 1.0 else output * scale


def scaled_hadamard_transform(
    u: torch.Tensor, scale: float
) -> torch.Tensor:
    """Hadamard transform with normalization fused when FHT is installed."""

    return _hadamard_transform(u, scale=scale)


def get_hadK(n, transpose=False):
    hadK, K = None, None
    if is_pow2(n):
        K = 1
        return hadK, K

    # load Hadamard matrices
    had_mat = torch.load(Path(__file__).parent / "had_mat.pt")
    k_list = list(had_mat.keys())
    k_list.sort()   # match smaller left-side Hadamard
    for k in k_list:
        if n % k == 0 and is_pow2(n // k):
            K, hadK = k, had_mat[k]
            hadK = hadK.T if transpose else hadK
            return hadK, K
        
    raise ValueError(f"Can't find appropriate Hadamard for size {n}!")


def random_orthogonal_matrix(size, device, generator=None):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64, generator=generator).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def random_hadamard_matrix(size, device, generator=None):
    Q = torch.randint(low=0, high=2, size=(size,), generator=generator).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.eye(size)
    return matmul_hadU(Q).to(device)


class HadamardTransform(torch.autograd.Function):
    """The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))"""

    @staticmethod
    def forward(u):
        return _hadamard_transform(u, scale=1.0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        # The transform is linear and has no saved-tensor state.  Defining
        # setup_context (and using the ctx-free forward signature) makes this
        # custom Function composable with torch.func transforms.
        pass

    @staticmethod
    def backward(ctx, grad):
        return _hadamard_transform(grad, scale=1.0)

    @staticmethod
    def jvp(ctx, grad_u):
        return _hadamard_transform(grad_u, scale=1.0)


def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()


def matmul_hadU_cuda(X, hadK, K):
    n = X.shape[-1]
    if K == 1:
        return scaled_hadamard_transform(
            X.contiguous(), scale=1.0 / math.sqrt(n)
        )
    # if transpose:
    #     hadK = hadK.T.contiguous()
    input = X.view(-1, K, n // K)
    input = scaled_hadamard_transform(
        input.contiguous(), scale=1.0 / math.sqrt(n)
    )
    input = hadK.to(device=input.device, dtype=input.dtype) @ input
    return input.reshape(X.shape)


def apply_exact_had_to_linear(module, had_dim=-1, output=False, R2=None):
    assert isinstance(module, torch.nn.Linear)
    in_features, out_features = module.in_features, module.out_features

    if had_dim != -1:
        assert is_pow2(had_dim), "Hadamard dimension must be a power of 2!"

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.to(device="cuda", dtype=torch.float32)

    if had_dim == -1:
        if output:
            had_K, K = get_hadK(out_features)
            W_ = matmul_hadU_cuda(W_.t(), had_K, K).t()
        if not output:
            had_K, K = get_hadK(in_features)
            W_ = matmul_hadU_cuda(W_, had_K, K)
    else:
        hadK = hadamard_matrix(had_dim, "cuda").to(torch.float64)
        if R2 is not None:
            hadK = R2.to(torch.float64)
        if output:
            W_ = W_.t()
            transposed_shape = W_.shape
            temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(transposed_shape).t()
        else:
            init_shape = W_.shape
            temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
            temp = temp.to(torch.float64) @ hadK
            W_ = temp.reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)
