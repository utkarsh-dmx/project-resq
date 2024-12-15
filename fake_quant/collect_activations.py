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
from mpl_toolkits.mplot3d import Axes3D

import os
from utils.utils import get_local_rank
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as fm
import seaborn as sns

log: Logger = utils.get_logger("resq", "collect_act.log")

FONTSIZE = 10
font_config = {"font.size": FONTSIZE, "font.family": "Times New Roman"}
plt.rcParams.update(font_config)
plt.rcParams["figure.figsize"] = (4.5, 4.5)


@torch.no_grad()
def plot_layer_benchmark():

    # Set Seaborn style
    sns.set(style="whitegrid")

    categories = [
        "[4096,4096]",
        "[8192,8192]",
        "[14336,4096]",
        "[4096,14336]",
        "[8192,28672]",
        "[28672,8192]",
    ]

    int4_benchmark_8192 = [1.35, 1.82, 1.67, 1.75, 2.36, 2.09]
    int4_benchmark_1024 = [1.25, 1.96, 1.68, 1.58, 2.23, 2.4]

    x = np.arange(len(categories))
    width = 0.35

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(
        x - width / 2,
        int4_benchmark_1024,
        width,
        label="seq_len : 1024",
        color="#4c72b0",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x + width / 2,
        int4_benchmark_8192,
        width,
        label="seq_len : 8192",
        color="#55a868",
        edgecolor="black",
    )

    # Adding data labels for each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    add_labels(bars1)
    add_labels(bars2)

    # Labels and Titles
    ax.set_xlabel("Layer dimensions", fontsize=18, fontweight="bold")
    ax.set_ylabel("Speedup", fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.legend(fontsize=14)

    # Grid and Aesthetics
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Show plot
    plt.tight_layout()
    plt.savefig("layer_benchmark.png", dpi=600)
    plt.clf()


@torch.no_grad()
def plot_rank_ablation():

    # Set Seaborn style
    sns.set(style="whitegrid")

    rank = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

    precision_4_downproj_4 = [8.8, 8.6, 8.55, 8.45, 8.33, 8.25, 8.18]
    precision_4_downproj_8 = [8.5, 8.31, 8.18, 8.07, 7.99, 7.92, 7.86]
    precision_3_downproj_8 = [13.57, 11.12, 9.88, 9.12, 8.63, 8.28, 7.99]

    fig, ax = plt.subplots(figsize=[5, 6])

    # Plot data with more distinct line styles and colors
    ax.plot(
        rank,
        precision_4_downproj_4,
        label="4/4/4, down_proj: INT4",
        marker="D",
        linestyle="-",
        linewidth=3,
        markersize=8,
    )

    ax.plot(
        rank,
        precision_4_downproj_8,
        label="4/4/4, down_proj: INT8",
        marker="o",
        linestyle="--",
        linewidth=3,
        markersize=8,
    )
    ax.plot(
        rank,
        precision_3_downproj_8,
        label="3/3/3, down_proj: INT8",
        marker="s",
        linestyle="-.",
        linewidth=3,
        markersize=8,
    )

    # Horizontal reference line
    ax.axhline(y=7.86, color="red", linestyle="--", linewidth=2)
    ax.text(rank[0], 7.86, "16-bit", color="red", fontsize=16, ha="left", va="bottom")

    # Add grid lines and labels
    ax.set_ylabel("Wikitext PPL", fontsize=14)
    ax.set_xlabel("r/channel_dimension", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # Add a title
    # ax.set_title("Precision vs Rank Comparison", fontsize=16, fontweight="bold")

    # Add legend with improved positioning
    ax.legend(fontsize=16, loc="upper right")

    # Save and show plot
    plt.tight_layout()
    plt.savefig("rank_ablation.png", dpi=600)
    plt.clf()


@torch.no_grad()
def plot_samples_ablation():

    # Set Seaborn style
    sns.set(style="whitegrid")

    samples = [16, 32, 64, 128, 256, 512]
    llama_3_8b_mmlu_calib_wiki = [0.563, 0.569, 0.569, 0.574, 0.574, 0.572]
    llama_3p2_3b_mmlu_calib_wiki = [0.488, 0.476, 0.478, 0.495, 0.474, 0.498]

    fig, ax = plt.subplots(figsize=[5, 6])

    # Plot data with more distinct line styles and colors
    ax.plot(
        samples,
        llama_3_8b_mmlu_calib_wiki,
        label="Llama-3-8b",
        marker="D",
        linestyle="-",
        color="#9467bd",
        linewidth=3,
        markersize=8,
    )

    ax.plot(
        samples,
        llama_3p2_3b_mmlu_calib_wiki,
        label="Llama-3.2-3b",
        marker="o",
        linestyle="--",
        linewidth=3,
        color="#d62728",
        markersize=8,
    )

    # Add grid lines and labels
    ax.set_ylabel("Avg. MMLU acc.", fontsize=14)
    ax.set_xlabel("nsamples", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # Add a title
    # ax.set_title("Precision vs Rank Comparison", fontsize=16, fontweight="bold")

    # Add legend with improved positioning
    ax.legend(fontsize=16, loc="center right")

    # Save and show plot
    plt.tight_layout()
    plt.savefig("nsamples_ablation.png", dpi=600)
    plt.clf()


@torch.no_grad()
def plot_layer_activations(act, save_file_name):
    # X = act.transpose(0, 1).abs().detach().numpy()
    if len(act.shape) == 3:
        # take the first batch
        act = act[0]

    act = act.float().detach().numpy()
    hidden_dim = act.shape[-1]

    fig = plt.figure(figsize=[4.5, 6.5])
    ax = fig.add_subplot(111)

    x = np.arange(hidden_dim)

    act_99 = np.percentile(act, 99, axis=0)
    act_1 = np.percentile(act, 1, axis=0)
    act_100 = act.max(axis=0)
    act_0 = act.min(axis=0)
    act_75 = np.percentile(act, 75, axis=0)
    act_50 = np.percentile(act, 25, axis=0)

    ax.fill_between(x, act_0, act_100, alpha=1.0, color="#38a4c8", label="min/max")
    ax.fill_between(
        x, act_1, act_99, alpha=1.0, color="#e76f51", label="1/99 percentile"
    )
    ax.fill_between(
        x, act_50, act_75, alpha=1.0, color="#f8e51d", label="25/75 percentile"
    )
    ax.set_xlabel("Channel", labelpad=0)
    ax.set_ylabel("Activation Value", labelpad=0)

    plt.savefig(save_file_name + "_1.png", dpi=600)
    plt.clf()

    act = np.abs(act)
    tokens, channels = act.shape
    x = np.arange(channels)
    y = np.arange(tokens)
    X, Y = np.meshgrid(x, y)
    # plt.plot(act2)

    # creating figure and 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # plotting the surface
    surf = ax.plot_surface(X, Y, act, cmap="coolwarm")

    ax.xaxis.set_tick_params(pad=-5)
    ax.yaxis.set_tick_params(pad=-3)
    ax.zaxis.set_tick_params(pad=-130)
    ax.zaxis.set_rotate_label(False)
    ax.view_init(elev=30, azim=120)
    # ax.axes.set_zlim3d(bottom=0, top=6)

    # Adding labels
    ax.set_xlabel("Channel", labelpad=0)
    ax.set_ylabel("Token", labelpad=0)

    plt.savefig(save_file_name + "_2.png")
    plt.clf()


def plot_mse_error(ptq_args):
    sns.set(style="whitegrid")

    save_file_name = os.path.join(ptq_args.output_dir, "mse_error_attn.png")
    save_path_attn = os.path.join(ptq_args.output_dir, "none", "attn_mse.pt")
    rotate_none = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "resq", "attn_mse.pt")
    rotate_resq = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quarot", "attn_mse.pt")
    rotate_quarot = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik", "attn_mse.pt")
    rotate_quik = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik_rotate", "attn_mse.pt")
    rotate_quik_rot = torch.load(save_path_attn).float().detach().numpy()

    lw = 2
    fig = plt.figure()
    plt.plot(rotate_none, label="INT4", lw=lw, color="grey")
    plt.plot(rotate_quarot, label="rotation", lw=lw, color="olive")
    plt.plot(rotate_quik, label="high precision outlier", lw=lw, color="tan")
    plt.plot(rotate_quik_rot, label="outlier+rot", lw=lw, color="blue")
    plt.plot(rotate_resq, label="ResQ", lw=lw, color="magenta")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("Quantization Error")
    plt.xlabel("Layer idx")

    plt.savefig(save_file_name)
    plt.clf()

    save_file_name = os.path.join(ptq_args.output_dir, "mse_error_mlp.png")

    save_path_attn = os.path.join(ptq_args.output_dir, "none", "mlp_mse.pt")
    rotate_none = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "resq", "mlp_mse.pt")
    rotate_resq = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quarot", "mlp_mse.pt")
    rotate_quarot = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik", "mlp_mse.pt")
    rotate_quik = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik", "mlp_mse.pt")
    rotate_quik = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik_rotate", "mlp_mse.pt")
    rotate_quik_rot = torch.load(save_path_attn).float().detach().numpy()

    lw = 2
    fig = plt.figure()
    plt.plot(rotate_none, label="INT4", lw=lw, color="grey")
    plt.plot(rotate_quarot, label="rotation", lw=lw, color="olive")
    plt.plot(rotate_quik, label="high precision outliers", lw=lw, color="tan")
    plt.plot(rotate_quik_rot, label="outlier+rot", lw=lw, color="blue")
    plt.plot(rotate_resq, label="ResQ", lw=lw, color="magenta")
    plt.yscale("log")
    plt.legend()
    plt.ylabel("Quantization Error")
    plt.xlabel("Layer idx")

    plt.savefig(save_file_name)
    plt.clf()

    ##############################
    save_file_name = os.path.join(ptq_args.output_dir, "snr_attn.png")
    save_path_attn = os.path.join(ptq_args.output_dir, "none", "attn_snr.pt")
    rotate_none = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "resq", "attn_snr.pt")
    rotate_resq = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quarot", "attn_snr.pt")
    rotate_quarot = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik", "attn_snr.pt")
    rotate_quik = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik_rotate", "attn_snr.pt")
    rotate_quik_rotate = torch.load(save_path_attn).float().detach().numpy()

    lw = 2
    fig, ax = plt.subplots(2, 1, figsize=[4.5, 7.5])

    ax[0].plot(rotate_none, label="INT4", lw=lw, color="grey")
    ax[0].plot(rotate_quarot, label="random rotation", lw=lw, color="#9467bd")
    ax[0].plot(rotate_quik, label="high precision outlier", lw=lw, color="#d62728")
    ax[0].plot(rotate_resq, label="ResQ", lw=lw, color="#e69f00")
    ax[0].plot(rotate_quik_rotate, label="outlier+rotation", lw=lw, color="#1f77b4")

    # ax[0].set_xlabel("Layer Idx")
    # ax[0].set_ylabel("Quantization SNR")
    # ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    save_path_attn = os.path.join(ptq_args.output_dir, "none", "mlp_snr.pt")
    rotate_none = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "resq", "mlp_snr.pt")
    rotate_resq = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quarot", "mlp_snr.pt")
    rotate_quarot = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik", "mlp_snr.pt")
    rotate_quik = torch.load(save_path_attn).float().detach().numpy()

    save_path_attn = os.path.join(ptq_args.output_dir, "quik_rotate", "mlp_snr.pt")
    rotate_quik_rotate = torch.load(save_path_attn).float().detach().numpy()

    ax[1].plot(rotate_none, label="baseline", lw=lw, color="grey")
    ax[1].plot(rotate_quarot, label="hadamard rotation", lw=lw, color="#9467bd")
    ax[1].plot(rotate_quik, label="high precision outlier", lw=lw, color="#d62728")
    ax[1].plot(rotate_resq, label="resq", lw=lw, color="#e69f00")
    ax[1].plot(rotate_quik_rotate, label="outlier+rotation", lw=lw, color="#1f77b4")

    # ax[1].set_xlabel("Layer Idx")
    # ax[1].set_ylabel("Quantization SNR")
    # ax[1].yaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    # plt.legend()

    plt.savefig(save_file_name, dpi=600)
    plt.clf()


@torch.no_grad()
def layerwise_mse(model_args, training_args, ptq_args, model, calib_data):
    mse_attn = []
    mse_mlp = []
    snr_mlp = []
    snr_attn = []

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
        R1_1 = R_dict["R1_1"].cuda().to(torch.float64)
        R1_2 = R_dict["R1_2"].cuda().to(torch.float64)
        R1 = torch.block_diag(R1_1, R1_2).cuda()

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
            error1 = inpq_k - dumped_inps["k_proj"]
            mse1 = (error1).pow(2).sum(-1).mean()
            snr1 = 10 * torch.log10(
                ((dumped_inps["k_proj"] ** 2).mean()) / ((error1**2).mean())
            )

            error2 = inpq_up - dumped_inps["gate_proj"]
            mse2 = (error2).pow(2).sum(-1).mean()
            snr2 = 10 * torch.log10(
                ((dumped_inps["gate_proj"] ** 2).mean()) / ((error2**2).mean())
            )
            mse_attn.append(mse1)
            snr_attn.append(snr1)
            mse_mlp.append(mse2)
            snr_mlp.append(snr2)

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
        snr_attn = torch.stack(snr_attn)
        snr_mlp = torch.stack(snr_mlp)

    return mse_attn, mse_mlp, snr_attn, snr_mlp


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

    # Rotate the weights
    if ptq_args.capture_layer_io:
        if ptq_args.rotate_mode != "none":
            fuse_norm_utils.fuse_layer_norms(model)
            if not (
                ptq_args.rotate_mode == "quarot" or ptq_args.rotate_mode == "spinquant"
            ):
                rotation_utils.fuse_basis_to_model(model, ptq_args)
            else:
                rotation_utils.rotate_model(model, ptq_args)
            if not (
                ptq_args.rotate_mode == "quarot" or ptq_args.rotate_mode == "spinquant"
            ):
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

        save_path = model_utils.get_layer_io_save_path(ptq_args)
        if not os.path.exists(save_path):
            # gotta collect activations for the layer
            dataset_ppl = eval_utils.evaluator(model, calib_data, utils.DEV, ptq_args)
            log.info("wiki2 ppl is: {}".format(dataset_ppl))
        else:
            logging.info(f"Activations dir already exists at : {save_path}")

        captured_io = torch.load(save_path)

        save_path_image = os.path.join(
            ptq_args.output_dir,
            "layer_io",
            ptq_args.rotate_mode,
            "attn_input",
            f"{ptq_args.layer_idx:03d}",
        )
        os.makedirs(os.path.dirname(save_path_image), exist_ok=True)
        acts = captured_io["input"]["k_proj"]
        plot_layer_activations(acts, save_path_image)

        save_path_image = os.path.join(
            ptq_args.output_dir,
            "layer_io",
            ptq_args.rotate_mode,
            "mlp_input",
            f"{ptq_args.layer_idx:03d}",
        )
        os.makedirs(os.path.dirname(save_path_image), exist_ok=True)

        acts = captured_io["input"]["gate_proj"]
        plot_layer_activations(acts, save_path_image)
        os.remove(save_path)

    if ptq_args.layerwise_mse:
        with torch.no_grad():
            # for rotate_mode in ["none", "resq", "quik", "quarot"]:
            # ptq_args.rotate_mode = rotate_mode
            # save_path_attn_mse = os.path.join(
            #     ptq_args.output_dir, ptq_args.rotate_mode, "attn_mse.pt"
            # )

            # save_path_attn_snr = os.path.join(
            #     ptq_args.output_dir, ptq_args.rotate_mode, "attn_snr.pt"
            # )

            # save_path_mlp_mse = os.path.join(
            #     ptq_args.output_dir, ptq_args.rotate_mode, "mlp_mse.pt"
            # )

            # save_path_mlp_snr = os.path.join(
            #     ptq_args.output_dir, ptq_args.rotate_mode, "mlp_snr.pt"
            # )

            # os.makedirs(os.path.dirname(save_path_attn_mse), exist_ok=True)
            # os.makedirs(os.path.dirname(save_path_attn_snr), exist_ok=True)
            # os.makedirs(os.path.dirname(save_path_mlp_mse), exist_ok=True)
            # os.makedirs(os.path.dirname(save_path_mlp_snr), exist_ok=True)

            # error_attn, error_mlp, snr_attn, snr_mlp = layerwise_mse(
            #     model_args, training_args, ptq_args, model, calib_data
            # )

            # torch.save(error_attn, save_path_attn_mse)
            # torch.save(snr_attn, save_path_attn_snr)
            # torch.save(error_mlp, save_path_mlp_mse)
            # torch.save(snr_mlp, save_path_mlp_snr)
            # print(
            #     f"Rotate Mode : {ptq_args.rotate_mode} error attn = {error_attn} error mlp = {error_mlp} snr attn = {snr_attn} snr mlp = {snr_mlp}"
            # )

            plot_mse_error(ptq_args)


if __name__ == "__main__":
    # collect_act()
    # plot_rank_ablation()
    # plot_samples_ablation()
    plot_layer_benchmark()
