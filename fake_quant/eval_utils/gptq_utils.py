# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import copy
import logging
import math
import pprint
import time

import torch
import torch.nn as nn
import tqdm

from utils import quant_utils, utils


class GPTQ:
    def __init__(self, layer, input_residual=False, residual_length=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.input_residual = input_residual  # keep 128 input channels FP
        self.residual_length = residual_length
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
    ):
        W_org = self.layer.weight.data.clone()
        W_org = W_org.float()
        W = W_org.clone()
        residual_dim = W_org.shape[-1] - self.residual_length
        if self.input_residual:
            if not self.quantizer.ready():
                self.quantizer.find_params(W[:, :residual_dim])
            if not self.input_residual_quantizer.ready():
                self.input_residual_quantizer.find_params(W[:, residual_dim:])

        else:
            if not self.quantizer.ready():
                self.quantizer.find_params(W)

        tick = time.time()
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        if static_groups:
            assert not self.input_residual  # not supported/tested yet.
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)]
                            )
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                if self.input_residual and i1 >= residual_dim:
                    q = self.input_residual_quantizer.quantize(w.unsqueeze(1)).flatten()
                else:
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning("NaN in weights")

            pprint.pprint(
                self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point
            )
            raise ValueError("NaN in weights")

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        utils.cleanup_memory(verbos=False)


@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    """
    From GPTQ repo
    """
    logging.info("-----GPTQ Quantization-----")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
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
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    quantizers = {}
    sequential = [
        [
            "self_attn.k_proj.module",
            "self_attn.v_proj.module",
            "self_attn.q_proj.module",
        ],
        ["self_attn.o_proj.module"],
        ["mlp.up_proj.module", "mlp.gate_proj.module"],
        ["mlp.down_proj.module"],
    ]

    sequential.append(["basis_change_1.module"])
    sequential.append(["basis_change_2.module"])

    model_dim = model.config.hidden_size
    residual_fraction = args.residual_fraction
    mlp_dim = model.config.intermediate_size

    for i in range(len(layers)):
        print(f"\nLayer {i}:", flush=True, end=" ")
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names if n in full.keys()}
            gptq = {}
            for name in subset:
                print(f"{name}", end="  ", flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not (args.w_asym)
                if "lm_head" in name:
                    layer_weight_bits = 16
                    continue
                if "basis_change" in name:
                    layer_weight_bits = 16  #####

                if args.int8_down_proj and "down_proj" in name:
                    layer_weight_bits = 8
                input_residual = False
                residual_length = 0

                if args.rotate_mode == "resq":
                    # mixed precision quantization for weights
                    if (
                        "k_proj" in name
                        or "q_proj" in name
                        or "v_proj" in name
                        or "up_proj" in name
                        or "gate_proj" in name
                        or "o_proj" in name
                    ):
                        residual_length = int(residual_fraction * model_dim)
                        input_residual = True
                gptq[name] = GPTQ(
                    subset[name],
                    input_residual=input_residual,
                    residual_length=residual_length,
                )
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits,
                    perchannel=True,
                    sym=layer_weight_sym,
                    mse=args.w_clip,
                )
                if gptq[name].input_residual:
                    gptq[name].input_residual_quantizer = quant_utils.WeightQuantizer()
                    gptq[name].input_residual_quantizer.configure(
                        8,
                        perchannel=True,
                        sym=layer_weight_sym,
                        mse=args.w_clip,
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=layer_w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                quantizers["model.layers.%d.%s" % (i, name)] = gptq[name].quantizer
                if input_residual:
                    quantizers["model.layers.%d.%s,input_residual" % (i, name)] = gptq[
                        name
                    ].input_residual_quantizer

                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    utils.cleanup_memory(verbos=True)
    logging.info("-----GPTQ Quantization Done-----\n")
    return quantizers


@torch.no_grad()
def rtn_fwrd(model, dev, args):
    """
    From GPTQ repo
    """
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    model_dim = model.config.hidden_size
    residual_fraction = args.residual_fraction

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if "lm_head" in name:
                layer_weight_bits = 16
                continue
            if "basis_change" in name:
                layer_weight_bits = 16  #####

            if args.int8_down_proj and "down_proj" in name:
                layer_weight_bits = 8

            residual_length = 0
            input_residual = False
            if args.rotate_mode == "resq":
                if (
                    "k_proj" in name
                    or "q_proj" in name
                    or "v_proj" in name
                    or "up_proj" in name
                    or "gate_proj" in name
                    or "o_proj" in name
                ):
                    residual_length = int(residual_fraction * model_dim)
                    input_residual = True

            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.w_clip,
            )
            if input_residual:
                input_residual_quantizer = quant_utils.WeightQuantizer()
                input_residual_quantizer.configure(
                    8,
                    perchannel=True,
                    sym=not (args.w_asym),
                    mse=args.w_clip,
                )
            W_org = subset[name].weight.data
            residual_dim = W_org.shape[-1] - residual_length
            if input_residual:
                W1 = W_org[:, :residual_dim]
                W2 = W_org[:, residual_dim:]
                quantizer.find_params(W1)
                input_residual_quantizer.find_params(W2)
                W1 = quantizer.quantize(W1)
                W2 = quantizer.quantize(W2)
                W_org = torch.cat([W1, W2], dim=-1)
            else:
                quantizer.find_params(W_org)
                quantizer.quantize(W_org)

            subset[name].weight.data = W_org.to(next(iter(layer.parameters())).dtype)

            quantizers["model.layers.%d.%s" % (i, name)] = quantizer.cpu()
            if input_residual:
                quantizers["model.layers.%d.%s,input_residual" % (i, name)] = (
                    input_residual_quantizer.cpu()
                )

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    utils.cleanup_memory(verbos=True)
    return quantizers