# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger
import logging

import torch
import torch.distributed as dist

from transformers import LlamaTokenizerFast, AutoConfig
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from eval_utils import rotation_utils
from utils import (
    data_utils,
    fuse_norm_utils,
    hadamard_utils,
    quant_utils,
    utils,
    model_utils,
)
from tqdm import tqdm

import os
from utils.utils import get_local_rank
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

log: Logger = utils.get_logger("resq")

FONTSIZE = 10

font_config = {"font.size": FONTSIZE, "font.family": "DejaVu Math TeX Gyre"}
plt.rcParams.update(font_config)
plt.rcParams["figure.figsize"] = (4, 4.5)


@torch.no_grad()
def plot_layer_activations(act, save_file_name):
    # X = act.transpose(0, 1).abs().detach().numpy()
    if len(act.shape) == 3:
        # take the first batch
        act = act[0]
    act = act.float().abs().detach().numpy()
    tokens, channels = act.shape

    x = np.arange(channels)
    y = np.arange(tokens)

    X, Y = np.meshgrid(x, y)

    # creating figure and 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # plotting the surface
    surf = ax.plot_surface(X, Y, act, cmap="coolwarm")

    ax.xaxis.set_tick_params(pad=-5)
    ax.yaxis.set_tick_params(pad=-3)
    ax.zaxis.set_tick_params(pad=-130)

    # Adding labels
    ax.set_xlabel("Channel", labelpad=0)
    ax.set_ylabel("Token", labelpad=0)

    plt.savefig(save_file_name)
    plt.clf()


@torch.no_grad()
def layerwise_mse(model_args, training_args, ptq_args, model, calib_data):
    mse_attn = []
    mse_mlp = []

    dev = utils.DEV

    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = calib_data.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = (
        input_ids[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)
    )  # (nsamples, seqlen)

    batch_size = ptq_args.bsz
    input_ids = [input_ids[i : i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
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
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    position_ids = cache["position_ids"]

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    rotate_flag = False
    if ptq_args.rotate_mode == "resq":
        R_dict = torch.load(ptq_args.optimized_rotation_path)
        R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
        R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
        R1 = torch.block_diag(R1_1, R1_2).cuda()
        U_cpk = torch.load(ptq_args.optimized_basis_path)
        U = torch.matmul(U_cpk["attn_mlp"].cuda(), R1).cuda()
        rotate_flag = True

    elif ptq_args.rotate_mode == "quarot":
        U = rotation_utils.get_orthogonal_matrix(
            model.config.hidden_size, "hadamard"
        ).cuda()
        rotate_flag = True
    elif ptq_args.rotate_mode == "quik":
        U_cpk = torch.load(ptq_args.optimized_basis_path)
        rotate_flag = True

        R_dict = torch.load(ptq_args.optimized_rotation_path)
        # R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
        # R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
        # R1 = torch.block_diag(R1_1, R1_2).cuda()

    with torch.no_grad():
        for i in tqdm(range(len(layers)), desc="(Evaluating MSE)  Layers"):
            if ptq_args.rotate_mode == "quik":
                key = f"layer.{i}.self_attn"
                # UA = torch.matmul(U_cpk[key].cuda(), R1)
                UA = U_cpk[key].cuda()
                key = f"layer.{i}.mlp"
                # UM = torch.matmul(U_cpk[key].cuda(), R1)
                UM = U_cpk[key].cuda()

            layer = layers[i].to(dev)

            # Dump the layer input and output
            captured_io = model_utils.capture_layer_io(
                layer, inps, attention_mask, position_ids, position_embeddings
            )
            dumped_inps = captured_io["input"]

            q1 = quant_utils.ActQuantizer()
            if ptq_args.rotate_mode == "quarot":
                residual_length = 0
            else:
                residual_length = int(
                    ptq_args.residual_fraction * model.config.hidden_size
                )

            q1.configure(
                bits=4,
                groupsize=ptq_args.a_groupsize,
                sym=not (ptq_args.a_asym),
                clip_ratio=ptq_args.a_clip_ratio,
                residual_length=residual_length,
                residual_bits=8,
            )
            inp_k = dumped_inps["k_proj"]
            inp_up = dumped_inps["gate_proj"]

            dtype = inp_k.dtype
            # rotate
            if rotate_flag:
                if ptq_args.rotate_mode == "quik":
                    inp_k = torch.matmul(inp_k.cuda(), UA.to(inp_k.dtype)).cpu()
                    inp_up = torch.matmul(inp_up.cuda(), UM.to(inp_up.dtype)).cpu()
                else:
                    inp_k = torch.matmul(inp_k.cuda(), U.to(inp_k.dtype)).cpu()
                    inp_up = torch.matmul(inp_up.cuda(), U.to(inp_up.dtype)).cpu()

            q1.find_params(inp_k)
            inpq_k = q1(inp_k)
            q1.free()

            q1.find_params(inp_up)
            inpq_up = q1(inp_up)
            q1.free()

            if rotate_flag:
                if ptq_args.rotate_mode == "quik":
                    inpq_up = torch.matmul(
                        inpq_up.cuda(), UM.t().to(inpq_up.dtype)
                    ).cpu()
                    inpq_k = torch.matmul(inpq_k.cuda(), UA.t().to(inpq_k.dtype)).cpu()

                else:
                    inpq_up = torch.matmul(
                        inpq_up.cuda(), U.t().to(inpq_up.dtype)
                    ).cpu()
                    inpq_k = torch.matmul(inpq_k.cuda(), U.t().to(inpq_k.dtype)).cpu()

            mse1 = (inpq_k - dumped_inps["k_proj"]).pow(2).sum(-1).mean()

            mse2 = (inpq_up - dumped_inps["gate_proj"]).pow(2).sum(-1).mean()
            mse_attn.append(mse1)
            mse_mlp.append(mse2)

            del inp_k, inp_up, inpq_up, inpq_k, captured_io, dumped_inps
            torch.cuda.empty_cache()

            for j in range(nbatches):
                outputs = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    #  defined.
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                outs[j] = outputs[0]
            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)

        model.lm_head = model.lm_head.to(dev)
        nlls = []
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        for i in range(nbatches):
            hidden_states = inps[i]
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :]
            shift_labels = input_ids[i][:, 1:]
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            neg_log_likelihood = loss.float().mean(dim=1)
            nlls.append(neg_log_likelihood)
        nlls_tensor = torch.cat(nlls)
        ppl = torch.exp(nlls_tensor.mean())
        model.config.use_cache = use_cache
        logging.info(f"\n WikiText2 PPL: {ppl.item():.3f}")

        mse_attn = torch.stack(mse_attn)
        mse_mlp = torch.stack(mse_mlp)

    return mse_attn, mse_mlp


def collect_act():
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    model_args, training_args, ptq_args = process_args_ptq()
    config = AutoConfig.from_pretrained(
        model_args.input_model,
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=dtype,
        config=config,
    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model.eval()
    # Rotate the weights
    if ptq_args.rotate_mode != "none":
        fuse_norm_utils.fuse_layer_norms(model)
        if not (
            ptq_args.rotate_mode == "quarot" or ptq_args.rotate_mode == "spinquant"
        ):
            rotation_utils.fuse_basis_to_model(model, ptq_args)
        else:
            rotation_utils.rotate_model(model, ptq_args)
        if not (ptq_args.quarot or ptq_args.spinquant):
            rotation_utils.rearrange_columns(model, ptq_args, False)

        utils.cleanup_memory(verbos=True)
        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                no_had = False
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = ptq_args.fp32_had
                qlayers[name].no_had = no_had

    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    model.seqlen = training_args.model_max_length

    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
    calib_data = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    if ptq_args.capture_layer_io:
        save_path = model_utils.get_layer_io_save_path(ptq_args)
        if not os.path.exists(save_path):
            # gotta collect activations for the layer
            eval_utils.evaluator(model, calib_data, utils.DEV, ptq_args)
        else:
            logging.info(f"Activations dir already exists at : {save_path}")

        captured_io = torch.load(save_path)

        save_path = os.path.join(
            ptq_args.output_dir,
            "layer_io",
            ptq_args.rotate_mode,
            f"{ptq_args.layer_idx:03d}.png",
        )
        acts = captured_io["input"]["k_proj"]
        plot_layer_activations(acts, save_path)

    if ptq_args.layerwise_mse:
        with torch.no_grad():
            # for rotate_mode in ["none", "resq", "quik", "quarot"]:
            # ptq_args.rotate_mode = rotate_mode
            save_path_attn = os.path.join(
                ptq_args.output_dir, ptq_args.rotate_mode, "attn.pt"
            )
            save_path_mlp = os.path.join(
                ptq_args.output_dir, ptq_args.rotate_mode, "mlp.pt"
            )

            os.makedirs(os.path.dirname(save_path_attn), exist_ok=True)
            os.makedirs(os.path.dirname(save_path_mlp), exist_ok=True)

            error_attn, error_mlp = layerwise_mse(
                model_args, training_args, ptq_args, model, calib_data
            )

            torch.save(error_attn, save_path_attn)
            torch.save(error_mlp, save_path_mlp)

            print(
                f"Rotate Mode : {ptq_args.rotate_mode} error attn = {error_attn} error mlp = {error_mlp}"
            )


if __name__ == "__main__":
    collect_act()
