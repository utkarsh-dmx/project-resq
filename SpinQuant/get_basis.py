# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    Trainer,
    default_data_collator,
)
from utils.data_utils import get_data
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
from train_utils.optimizer import SGDG
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict
from utils import utils, data_utils, fuse_norm_utils, quant_utils
from tqdm import tqdm
import transformers
from utils.hadamard_utils import random_orthogonal_matrix
import time

log: Logger = get_logger("resq", "get_basis.log")


@torch.no_grad()
def get_outlier_rotations(model_args, training_args, ptq_args) -> None:

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=torch.float32,
    )
    model.seqlen = training_args.model_max_length

    transformers.set_seed(ptq_args.seed)
    model.eval()

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)

    utils.cleanup_memory(verbos=True)

    log.info("Model init completed for training {}".format(model))
    log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    seqlen = model.seqlen
    train_data = data_utils.get_data(
        seed=ptq_args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        eval_mode=False,
        nsamples=ptq_args.nsamples,
        calib_dataset=ptq_args.calib_dataset,
    )

    nbatches = len(train_data)
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(utils.DEV)
    model.model.rotary_emb = model.model.rotary_emb.to(utils.DEV)
    # model.R1.weight.data.copy_() = model.R1.to(utils.DEV)

    layers[0] = layers[0].to(utils.DEV)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=utils.DEV,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nbatches):
        batch = train_data[i][0].to(utils.DEV)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()

    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches

    basis_dict = {}
    rotation_dict = {}
    attention_mask = cache["attention_mask"]

    hidden_dim = model.config.hidden_size
    head_dim = hidden_dim // model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    down_proj_blocksize = ptq_args.down_proj_blocksize
    nlayers = len(layers)
    residual_fraction = ptq_args.residual_fraction
    residual_length_hidden = int(residual_fraction * hidden_dim)
    residual_length_head = int(residual_fraction * head_dim)
    residual_length_down_proj = int(residual_fraction * down_proj_blocksize)
    rotation_granularity = ptq_args.rotation_granularity

    for i in tqdm(range(nlayers), desc="(Collecting Outliers) Layers"):
        layer = layers[i].to(utils.DEV)

        hooks = []

        input_up_proj = []
        output_vproj = []
        output_kpos = []
        input_qkv_proj = []

        def hook_fn_upproj(module, input, output):
            input_up_proj.append(input[0].cpu())

        def hook_fn_vproj(module, input, output):
            global out_vp
            out_vp = output
            output_vproj.append(output.cpu())

        def hook_fn_kproj(module, input, output):
            global output_kproj
            output_kproj = output

        def hook_fn_qproj(module, input, output):
            global output_qproj
            output_qproj = output
            input_qkv_proj.append(input[0].cpu())

        hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn_upproj))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn_kproj))
        hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj))

        for j in range(nbatches):
            with torch.no_grad():
                # register forward hook for getting inputs to MLP

                # 1 sample at a time
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]

                # rope cos, sin
                cos, sin = layer.self_attn.rotary_emb(out_vp, position_ids)

                # reshape to get key states per head
                key_states = output_kproj.view(
                    1,
                    seqlen,
                    layer.self_attn.num_key_value_heads,
                    layer.self_attn.head_dim,
                ).transpose(1, 2)
                # key_states = repeat_kv(key_states, layer.self_attn.num_key_value_groups)

                # reshape to get query states per head
                query_states = output_qproj.view(
                    1,
                    seqlen,
                    layer.self_attn.num_heads,
                    layer.self_attn.head_dim,
                ).transpose(1, 2)

                # apply rotary embedding
                query_states_pos, key_states_pos = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
                output_kpos.append(key_states_pos.cpu())

        for hook in hooks:
            hook.remove()
        # reshape to get value states per head
        value_states = torch.stack(output_vproj).view(
            -1,
            layer.self_attn.num_key_value_heads,
            layer.self_attn.head_dim,
        )

        input_qkv_proj = torch.stack(input_qkv_proj).view(-1, hidden_dim)
        input_up_proj = torch.stack(input_up_proj).view(-1, hidden_dim)
        output_kpos = torch.stack(output_kpos).view(
            -1,
            layer.self_attn.num_key_value_heads,
            layer.self_attn.head_dim,
        )

        layers[i] = layers[i].cpu()

        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # find outlier channels
        basis_dict["layer." + str(i) + ".mlp"] = torch.eye(
            hidden_dim, dtype=torch.float64
        )[torch.sort(input_up_proj.abs().sum(0), descending=False).indices.cpu()].t()
        basis_dict["layer." + str(i) + ".self_attn"] = (
            torch.eye(hidden_dim, dtype=torch.float64)[
                torch.sort(input_qkv_proj.abs().sum(0), descending=False).indices.cpu()
            ]
            .cpu()
            .t()
        )

        basis_v_proj = []
        basis_k_pos = []
        for j in range(layer.self_attn.num_key_value_heads):
            basis_v_proj.append(
                torch.eye(layer.self_attn.head_dim, dtype=torch.float64)[
                    torch.sort(value_states[:, j].abs().sum(0), descending=False)
                    .indices.cpu()
                    .t()
                ]
            )

            basis_k_pos.append(
                torch.eye(layer.self_attn.head_dim, dtype=torch.float64)[
                    torch.sort(output_kpos[:, j].abs().sum(0), descending=False)
                    .indices.cpu()
                    .t()
                ]
            )

        basis_dict["layer." + str(i) + ".self_attn.value"] = torch.stack(
            basis_v_proj
        ).cpu()
        basis_dict["layer." + str(i) + ".self_attn.key_pos"] = torch.stack(
            basis_k_pos
        ).cpu()

    torch.cuda.empty_cache()
    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    path = os.path.join(
        model_args.output_rotation_path,
        "U-" + "outliers-" + model_args.input_model.split("/")[1] + ".bin",
    )

    torch.save(
        basis_dict,
        path,
    )


@torch.no_grad()
def get_basis(model_args, training_args, ptq_args) -> None:

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=torch.float32,
    )
    model.seqlen = training_args.model_max_length
    transformers.set_seed(ptq_args.seed)
    model.eval()

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)

    utils.cleanup_memory(verbos=True)

    log.info("Model init completed for training {}".format(model))
    log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    seqlen = model.seqlen
    train_data = data_utils.get_data(
        seed=ptq_args.seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        eval_mode=False,
        nsamples=ptq_args.nsamples,
        calib_dataset=ptq_args.calib_dataset,
    )
    nbatches = len(train_data)
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(utils.DEV)
    model.model.rotary_emb = model.model.rotary_emb.to(utils.DEV)
    # model.R1.weight.data.copy_() = model.R1.to(utils.DEV)

    layers[0] = layers[0].to(utils.DEV)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=utils.DEV,
    )
    inps = [0] * nbatches
    cache = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nbatches):
        batch = train_data[i][0].to(utils.DEV)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()

    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches

    basis_dict = {}
    rotation_dict = {}
    attention_mask = cache["attention_mask"]

    hidden_dim = model.config.hidden_size
    head_dim = hidden_dim // model.config.num_attention_heads
    kv_heads = model.config.num_key_value_heads
    down_proj_blocksize = ptq_args.down_proj_blocksize
    nlayers = len(layers)
    residual_fraction = ptq_args.residual_fraction
    residual_length_hidden = int(residual_fraction * hidden_dim)
    residual_length_head = int(residual_fraction * head_dim)
    residual_length_down_proj = int(residual_fraction * down_proj_blocksize)
    rotation_granularity = ptq_args.rotation_granularity

    # initialize covariance matrices
    H_attn = torch.zeros((len(layers), hidden_dim, hidden_dim), device=utils.DEV)
    H_mlp = torch.zeros((len(layers), hidden_dim, hidden_dim), device=utils.DEV)
    H_down_proj = torch.zeros(
        (
            nlayers,
            down_proj_blocksize,
            down_proj_blocksize,
        ),
        device=utils.DEV,
    )  # block down_proj with certain dimension
    H_value = torch.zeros(
        (
            nlayers,
            model.config.num_key_value_heads,
            head_dim,
            head_dim,
        ),
        device=utils.DEV,
    )
    H_key_pos = torch.zeros(
        (
            nlayers,
            kv_heads,
            head_dim,
            head_dim,
        ),
        device=utils.DEV,
    )
    rotation_dict["R1_1"] = random_orthogonal_matrix(
        hidden_dim - residual_length_hidden, "cuda"
    )
    rotation_dict["R1_2"] = random_orthogonal_matrix(residual_length_hidden, "cuda")

    rotation_dict["R2_1"] = random_orthogonal_matrix(
        head_dim - residual_length_head,
        "cuda",
    )
    rotation_dict["R2_2"] = random_orthogonal_matrix(residual_length_head, "cuda")

    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    rotation_path = os.path.join(
        model_args.output_rotation_path,
        "R-"
        + str(residual_fraction)
        + "-"
        + model_args.input_model.split("/")[1]
        + ".bin",
    )
    if not os.path.exists(rotation_path):
        torch.save(
            rotation_dict,
            rotation_path,
        )
    basis_path = os.path.join(
        model_args.output_rotation_path,
        "U-"
        + str(ptq_args.calib_dataset)
        + "-"
        + str(ptq_args.nsamples)
        + "-"
        + model_args.input_model.split("/")[1]
        + ".bin",
    )
    if not os.path.exists(basis_path):
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        for i in tqdm(range(nlayers), desc="(Collecting Cov matrices) Layers"):
            layer = layers[i].to(utils.DEV)

            hooks = []

            def hook_fn_upproj(module, input, output):
                global input_up_proj
                input_up_proj = input[0]

            def hook_fn_vproj(module, input, output):
                global output_vproj
                output_vproj = output

            def hook_fn_kproj(module, input, output):
                global output_kproj
                output_kproj = output

            def hook_fn_qproj(module, input, output):
                global output_qproj, input_qkv_proj
                output_qproj = output
                input_qkv_proj = input[0]

            def hook_fn_downproj(module, input, output):
                global input_down_proj
                input_down_proj = input[0]

            hooks.append(layer.mlp.up_proj.register_forward_hook(hook_fn_upproj))
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn_downproj))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_fn_vproj))
            hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_fn_kproj))
            hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_fn_qproj))

            for j in range(nbatches):
                with torch.no_grad():
                    # register forward hook for getting inputs to MLP

                    # 1 sample at a time
                    outs[j] = layer(
                        inps[j],
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]

                    # reshape to get value states per head
                    value_states = output_vproj.view(
                        1,
                        seqlen,
                        layer.self_attn.num_key_value_heads,
                        layer.self_attn.head_dim,
                    ).transpose(1, 2)

                    # rope cos, sin
                    cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)

                    # reshape to get key states per head
                    key_states = output_kproj.view(
                        1,
                        seqlen,
                        layer.self_attn.num_key_value_heads,
                        layer.self_attn.head_dim,
                    ).transpose(1, 2)
                    # key_states = repeat_kv(key_states, layer.self_attn.num_key_value_groups)

                    # reshape to get query states per head
                    query_states = output_qproj.view(
                        1,
                        seqlen,
                        layer.self_attn.num_heads,
                        layer.self_attn.head_dim,
                    ).transpose(1, 2)

                    # apply rotary embedding
                    query_states_pos, key_states_pos = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin
                    )

                    # calculate covariance
                    H_mlp[i] += torch.sum(
                        input_up_proj.double().mT @ input_up_proj.double(), dim=0
                    )  # shape : [hidden_dim, hidden_dim]

                    H_attn[i] += torch.sum(
                        input_qkv_proj.double().mT @ input_qkv_proj.double(), dim=0
                    )  # shape : [hidden_dim, hidden_dim]
                    H_value[i] += torch.sum(
                        value_states.double().mT @ value_states.double(), dim=0
                    )  # shape : [num_heads, head_dim, head_dim]

                    H_key_pos[i] += torch.sum(
                        key_states_pos.double().mT @ key_states_pos.double(), dim=(0)
                    )  # shape : [num_kv_heads, head_dim, head_dim]

                    H_down_proj[i] += torch.sum(
                        input_down_proj.view(
                            input_down_proj.shape[0], -1, ptq_args.down_proj_blocksize
                        )
                        .double()
                        .mT
                        @ input_down_proj.view(
                            input_down_proj.shape[0], -1, ptq_args.down_proj_blocksize
                        ).double(),
                        dim=(0),
                    )  # shape : [1024, 1024]
            for hook in hooks:
                hook.remove()

            layers[i] = layers[i].cpu()

            torch.cuda.empty_cache()

            inps, outs = outs, inps

        if "per_layer" in rotation_granularity.lower():
            for i in tqdm(range(nlayers), desc="(Getting Basis) Layers"):

                # eigen decomposition of attn
                eval_attn, evec_attn = perform_eigen_decomp(H_attn[i] / nbatches)

                # eigen decomposition of up proj
                eval_mlp, evec_mlp = perform_eigen_decomp(H_mlp[i] / nbatches)

                # eigen decomposition of down proj
                eval_down_proj, evec_down_proj = perform_eigen_decomp(
                    H_down_proj[i] / nbatches
                )

                # eigen decomposition of value states
                eval_value, evec_value = perform_eigen_decomp(
                    H_value[i] / nbatches, per_head=True, num_heads=H_value.shape[0]
                )

                # eigen decomposition of key states after rope embedding
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    H_key_pos[i] / nbatches, per_head=True, num_heads=H_key_pos.shape[0]
                )

                print(
                    "up proj:",
                    (eval_mlp[0:100].sum() / eval_mlp.sum()).item(),
                    "down proj:",
                    (eval_down_proj[0:100].sum() / eval_down_proj.sum()).item(),
                    ", hidden_attn:",
                    (eval_attn[0:100].sum() / eval_attn.sum()).item(),
                    ", v_proj:",
                    (eval_value[:, :32].sum(1) / eval_value.sum(1)).mean().item(),
                    # ", kq_proj:",
                    # (eval_kq[:, :32].sum(1) / eval_kq.sum(1)).mean().item(),
                    ", k_proj_pos:",
                    (eval_k_pos[:, :32].sum(1) / eval_k_pos.sum(1)).mean().item(),
                )

                basis_dict["config"] = "per_layer_rotation"
                basis_dict["layer." + str(i) + ".mlp"] = evec_mlp.cpu()
                basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()
                basis_dict["layer." + str(i) + ".self_attn"] = evec_attn.cpu()
                basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
                basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()

        elif "full_shared" in rotation_granularity.lower():
            # eigen decomposition of attn and mlp
            eval_attn_mlp, evec_attn_mlp = perform_eigen_decomp(
                (H_attn.sum(0) + H_mlp.sum(0)) / (2 * nbatches * nlayers)
            )

            basis_dict["config"] = "full_shared_rotation"
            basis_dict["attn_mlp"] = evec_attn_mlp.cpu()

            for i in tqdm(range(nlayers), desc="(Getting Rotations) Layers"):
                # eigen decomposition of down proj
                eval_down_proj, evec_down_proj = perform_eigen_decomp(
                    H_down_proj[i] / nbatches
                )
                # eigen decomposition of value states
                eval_value, evec_value = perform_eigen_decomp(
                    H_value[i] / nbatches, per_head=True, num_heads=kv_heads
                )
                # eigen decomposition of key states after rope embedding
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    H_key_pos[i].sum(0) / (kv_heads * nbatches),
                )

                basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
                basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()
                basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()

                print(
                    "down proj:",
                    (
                        eval_down_proj[-residual_length_down_proj:].sum()
                        / eval_down_proj.sum()
                    ).item(),
                    ", hidden_attn_mlp:",
                    (
                        eval_attn_mlp[-residual_length_hidden:].sum()
                        / eval_attn_mlp.sum()
                    ).item(),
                    ", v_proj:",
                    (eval_value[:, -residual_length_head:].sum(1) / eval_value.sum(1))
                    .mean()
                    .item(),
                    ", k_proj_pos:",
                    (eval_k_pos[-residual_length_head:].sum() / eval_k_pos.sum())
                    .mean()
                    .item(),
                )

        elif "one_per_decoder" in rotation_granularity.lower():

            for i in tqdm(range(nlayers), desc="(Getting Basis) Layers"):
                # eigen decomposition of attn and mlp
                eval_attn_mlp, evec_attn_mlp = perform_eigen_decomp(
                    (H_attn[i] + H_mlp[i]) / (2 * nbatches)
                )
                # eigen decomposition of down proj
                eval_down_proj, evec_down_proj = perform_eigen_decomp(
                    H_down_proj[i] / nbatches
                )

                # eigen decomposition of value states
                eval_value, evec_value = perform_eigen_decomp(
                    H_value[i] / nbatches, per_head=True, num_heads=kv_heads
                )

                # eigen decomposition of key states after rope embedding
                eval_k_pos, evec_k_pos = perform_eigen_decomp(
                    H_key_pos[i] / nbatches, per_head=True, num_heads=kv_heads
                )

                print(
                    "down proj:",
                    (eval_down_proj[0:100].sum() / eval_down_proj.sum()).item(),
                    ", attn_mlp:",
                    (eval_attn_mlp[0:100].sum() / eval_attn_mlp.sum()).item(),
                    ", v_proj:",
                    (eval_value[:, :32].sum(1) / eval_value.sum(1)).mean().item(),
                    # ", kq_proj:",
                    # (eval_kq[:, :32].sum(1) / eval_kq.sum(1)).mean().item(),
                    ", k_proj_pos:",
                    (eval_k_pos[:, :32].sum(1) / eval_k_pos.sum(1)).mean().item(),
                )

                basis_dict["config"] = "one_per_decoder"
                basis_dict["layer." + str(i) + ".mlp.down_proj"] = evec_down_proj.cpu()
                basis_dict["layer." + str(i) + ".self_attn_mlp"] = evec_attn_mlp.cpu()
                basis_dict["layer." + str(i) + ".self_attn.value"] = evec_value.cpu()
                basis_dict["layer." + str(i) + ".self_attn.key_pos"] = evec_k_pos.cpu()

        torch.cuda.empty_cache()

        torch.save(
            basis_dict,
            basis_path,
        )
    else:
        print(f"Basis rotations already exist at {basis_path}")


def perform_eigen_decomp(Cov_matrix, per_head=False, num_heads=0):
    # performs eigen decomposition and returns
    # the sorted eigen values and eigen vectors
    if per_head:
        assert num_heads != 0  # cannot use per head and not pass num_heads
        eval = []
        evec = []
        for hd in range(num_heads):
            H = Cov_matrix[hd]
            damp = 0.01 * torch.mean(torch.diag(H))
            diag = torch.arange(H.shape[-1]).to(device=H.device)
            H[diag, diag] = H[diag, diag] + damp
            X = torch.linalg.eigh(H.to(torch.float64))
            index = torch.argsort(X[0])
            eval.append(X[0][index])
            evec.append(X[1][:, index])
        eval = torch.stack(eval)
        evec = torch.stack(evec)
    else:
        H = Cov_matrix
        damp = 0.01 * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[-1]).to(device=H.device)
        H[diag, diag] = H[diag, diag] + damp
        X = torch.linalg.eigh(H.to(torch.float64))
        index = torch.argsort(X[0])
        eval = X[0][index]
        evec = X[1][:, index]

    return eval, evec


if __name__ == "__main__":
    model_args, training_args, ptq_args = process_args_ptq()
    if ptq_args.rotate_mode == "quik":
        get_outlier_rotations(model_args, training_args, ptq_args)
    else:
        get_basis(model_args, training_args, ptq_args)
