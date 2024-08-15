# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
    matmul_hadU_cuda,
    get_hadK,
)
from utils.utils import HadamardTransform
from utils.utils import get_local_rank


def random_orthogonal_matrix(size, device):
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
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device="cuda"):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
    # Rotate the embeddings.
    for W in [model.model.embed_tokens]:
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
    # Rotate output matrix of the self-attention layer.
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
    # Rotate the MLP input weights.
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
        W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1, R4, skip_R4=False):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
    if not skip_R4:
        if R4 is not None:
            device = W.weight.device
            # 172 * torch.inverse(R4
            W.weight.data = matmul_hadU_cuda(
                W.weight.data.float().cuda(),
                R4.shape[0] * torch.inverse(R4).t(),
                R4.shape[0],
            ).to(device=device, dtype=dtype)
        else:
            apply_exact_had_to_linear(
                W, had_dim=-1, output=False
            )  # apply exact (inverse) hadamard on the weights of mlp output
    if W.bias is not None:
        b = W.bias.data.to(device="cuda", dtype=torch.float64)
        W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = model.lm_head
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_basis_change(layer, R1: torch.Tensor) -> None:
    # Rotate the head.
    W = layer.basis_change_1
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    # W_ = torch.eye(4096, device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, torch.matmul(W_, R1)).to(
        device="cpu", dtype=dtype
    )

    W = layer.basis_change_2
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    # W_ = torch.eye(4096, device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(R1.T, torch.matmul(W_, R1)).to(
        device="cpu", dtype=dtype
    )


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


def create_orthogonal(rot):
    """
    helper function to convert any random matrix to orthogonal matrix. Leveraging Cayley transform
        1. Convert input A to a skew symmetric matrix by sk_sm = (A - A.t())/2
        2. Use cayley transform to convert skew symmetric matrix to orthogonal matrix orth = (I-A)((I+A)^-1)
    """
    dtype = rot.dtype
    device = rot.device
    sk_sm = 0.5 * (rot - rot.t())
    eye = torch.eye(rot.shape[0], device=device, dtype=dtype)
    p1 = eye - sk_sm
    p2 = torch.inverse((eye + sk_sm).to(torch.float32)).to(dtype)
    orth = torch.matmul(p1, p2)
    orth = orth / orth.norm(p=2, dim=0)

    return orth


@torch.inference_mode()
def rotate_model(model, args):
    R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
    # R1_1 = random_hadamard_matrix(128, "cuda")
    # R1_2 = random_orthogonal_matrix(model.config.hidden_size - 128, "cuda")
    # R1 = torch.block_diag(R1_1, R1_2)
    if args.optimized_rotation_path is not None:
        R_cpk = args.optimized_rotation_path
        R1_1 = torch.load(R_cpk)["R1_1"].cuda().to(torch.float64)
        R1_2 = torch.load(R_cpk)["R1_2"].cuda().to(torch.float64)
        # R1 = torch.kron(R1_2, R1_1)
        R1 = torch.block_diag(R1_1, R1_2)
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    rotate_embeddings(model, R1)
    rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        if args.optimized_rotation_path is not None:
            key = f"model.layers.{idx}.self_attn.R2"
            R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
            key = f"model.layers.{idx}.mlp.R4"
            R4 = torch.load(R_cpk)[key].cuda().to(torch.float64)
            # R4 = Non
        else:
            R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
            R4 = None
        rotate_attention_inputs(layers[idx], R1)
        rotate_attention_output(layers[idx], R1)
        rotate_mlp_input(layers[idx], R1)
        rotate_mlp_output(layers[idx], R1, R4)
        rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)
        rotate_basis_change(layers[idx], R1)


@torch.inference_mode()
def fuse_basis_to_model(model, args):
    U_cpk = torch.load("./rotation/U.bin")

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    dev = model.model.embed_tokens.weight.device
    U_attn = U_cpk["layer.0.self_attn"].cuda()

    local_rank = get_local_rank()
    # if local_rank == 0:
    #     breakpoint()
    torch.distributed.barrier()
    rotate_embeddings(model, U_attn)
    # rotate_head(model, R1)
    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Fusing Basis")):
        key = f"layer.{idx}.self_attn"
        U_attn = U_cpk[key].cuda()
        rotate_attention_inputs(layers[idx], U_attn)

        key = f"layer.{idx}.mlp"
        U_mlp = U_cpk[key].cuda()
        rotate_attention_output(layers[idx], U_mlp)
        rotate_mlp_input(layers[idx], U_mlp)
        dtype = layer.basis_change_1.weight.dtype
        layer.basis_change_1.weight.data.copy_(
            torch.matmul(U_mlp.T, U_attn).to(device="cpu", dtype=dtype)
        )
        # rotate MLP output for every layer except last layer
        if idx == model.config.num_hidden_layers - 1:
            dtype = layer.basis_change_2.weight.dtype
            W_rot_change = U_mlp.to(device="cpu", dtype=dtype)
            layer.basis_change_2.weight.data.copy_(W_rot_change)
        else:
            key = f"layer.{idx+1}.self_attn"
            U_attn_next = U_cpk[key].cuda()
            rotate_mlp_output(layers[idx], U_attn_next, R4=None, skip_R4=True)

            dtype = layer.basis_change_2.weight.dtype
            W_rot_change = torch.matmul(U_attn_next.T, U_mlp).to(
                device="cpu", dtype=dtype
            )
            layer.basis_change_2.weight.data.copy_(W_rot_change)

    # return model


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs["k_groupsize"] in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for K-cache"
            self.k_bits = kwargs["k_bits"]
            self.k_groupsize = kwargs["k_groupsize"]
            self.k_sym = kwargs["k_sym"]
            self.k_clip_ratio = kwargs["k_clip_ratio"]
            self.k_quantizer.configure(
                bits=self.k_bits,
                groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                sym=self.k_sym,
                clip_ratio=self.k_clip_ratio,
            )

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)
            self.k_quantizer.find_params(token_wise_k)
            k = (
                self.k_quantizer(token_wise_k)
                .reshape((bsz, seq_len, num_heads, head_dim))
                .transpose(1, 2)
                .to(q)
            )
        else:  # head-wise quantization
            per_head_k = k.view(-1, head_dim)
            self.k_quantizer.find_params(per_head_k)
            k = (
                self.k_quantizer(per_head_k)
                .reshape((bsz, num_heads, seq_len, head_dim))
                .to(q)
            )

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
