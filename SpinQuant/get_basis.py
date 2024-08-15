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
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
from utils.data_utils import get_wikitext2
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import random_hadamard_matrix, get_hadK
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict
from utils import utils, data_utils, fuse_norm_utils, quant_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import transformers
from optimize_rotation import RotateModule

log: Logger = get_logger("spinquant")


@torch.no_grad()
def train() -> None:
    model_args, training_args, ptq_args = process_args_ptq()

    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=torch.float32,
    )
    model.seqlen = training_args.model_max_length

    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))

    transformers.set_seed(ptq_args.seed)
    model.eval()

    # Rotate the weights
    fuse_norm_utils.fuse_layer_norms(model)

    # R1 = torch.eye(
    #     model.config.hidden_size, device="cuda"
    # )  # kept identity to not break code. doesn't have any effect
    # model.R1 = RotateModule(R1)
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
    train_data = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        eval_mode=False,
        nsamples=512,
    )
    nbatches = len(train_data)
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(utils.DEV)
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
    position_ids = cache["position_ids"]

    torch.cuda.empty_cache()
    # inps = torch.stack(inps).squeeze()
    outs = [0] * nbatches

    basis_dict = {}
    attention_mask = cache["attention_mask"]
    for i in tqdm(range(len(layers)), desc="(Getting Basis) Layers"):
        layer = layers[i].to(utils.DEV)
        input_mlp = []

        def hook_fn(module, input, output):
            # input_mlp.append(input[0])
            input_mlp.append(input[0])

        hook_handle = layer.mlp.up_proj.register_forward_hook(hook_fn)
        H_attn = 0.0
        H_mlp = 0.0
        for j in range(nbatches):
            with torch.no_grad():
                # register forward hook for getting inputs to MLP

                # 1 sample at a time
                outs[j] = layer(
                    inps[j],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    R1=None,
                )[0]

                # calculate covariance
                H_mlp += torch.sum(
                    input_mlp[0].double().mT @ input_mlp[0].double(), dim=0
                )
                H_attn += torch.sum(inps[j].double().mT @ inps[j].double(), dim=0)
                input_mlp = []
        hook_handle.remove()

        torch.cuda.empty_cache()

        # eigen decomposition of attn
        X_eig_attn = torch.linalg.eigh(H_attn)
        index = torch.argsort(X_eig_attn[0], descending=True)
        eval_attn = X_eig_attn[0][index]
        evec_attn = X_eig_attn[1][:, index]
        del H_attn, X_eig_attn

        # eigen decomposition of mlp
        X_eig_mlp = torch.linalg.eigh(H_mlp)
        index = torch.argsort(X_eig_mlp[0], descending=True)
        eval_mlp = X_eig_mlp[0][index]
        evec_mlp = X_eig_mlp[1][:, index]
        del H_mlp, X_eig_mlp

        del layer

        print(
            "mlp:",
            (eval_mlp[0:100].sum() / eval_mlp.sum()).item(),
            ", attn:",
            (eval_attn[0:100].sum() / eval_attn.sum()).item(),
        )
        basis_dict["layer." + str(i) + ".mlp"] = evec_mlp.cpu()
        basis_dict["layer." + str(i) + ".self_attn"] = evec_attn.cpu()

        torch.cuda.empty_cache()

        inps, outs = outs, inps

    os.makedirs(model_args.output_rotation_path, exist_ok=True)
    path = os.path.join(model_args.output_rotation_path, "U.bin")
    torch.save(
        basis_dict,
        path,
    )
    # also remove the checkpoint folder, we dont need it not
    # os.rmdir(training_args.output_dir)


if __name__ == "__main__":
    train()
