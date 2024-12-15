# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils
from utils.hadamard_utils import (
    random_orthogonal_matrix,
)


def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    # Rotate the weights
    if not args.rotate_mode == "none":
        fuse_norm_utils.fuse_layer_norms(model)
        if not (args.rotate_mode == "quarot" or args.rotate_mode == "spinquant"):
            rotation_utils.fuse_basis_to_model(model, args)
        else:
            rotation_utils.rotate_model(model, args)
        if not (args.rotate_mode == "quarot" or args.rotate_mode == "spinquant"):
            rotation_utils.rearrange_columns(model, args, False)

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
                qlayers[name].fp32_had = args.fp32_had
                qlayers[name].no_had = no_had

    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])

        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )
        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads
            mlp_dim = model.config.intermediate_size
            v_groupsize = head_dim
            if args.rotate_mode == "quarot" or args.rotate_mode == "spinquant":
                residual_length = 0
            else:
                residual_fraction = args.residual_fraction
                residual_length = int(residual_fraction * model_dim)

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                if args.rotate_mode == "quarot" or args.rotate_mode == "spinquant":
                    v_residual_length = 0
                else:
                    # per group residual
                    v_residual_length = int(v_groupsize * residual_fraction)
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                    residual_length=v_residual_length,
                    residual_bits=8,
                )
            if "o_proj" in name:
                layer_groupsize = head_dim
                if args.rotate_mode == "quarot" or args.rotate_mode == "spinquant":
                    residual_length = 0
                else:
                    # per group residual
                    residual_length = int(v_groupsize * residual_fraction)

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "basis_change" in name:
                layer_input_bits = 8  #####

            if "down_proj" in name:  # Set the down_proj precision
                residual_length = 0
                if args.int8_down_proj:
                    layer_input_bits = 8

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                residual_length=residual_length,
                residual_bits=8,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            if args.rotate_mode == "quarot" or args.rotate_mode == "spinquant":
                residual_length_k = 0
            else:
                U_cpk = torch.load(args.optimized_basis_path)
                residual_length_k = int(args.residual_fraction * head_dim)
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_bits_residual": 8,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
                "residual_length_k": residual_length_k,
            }

            for idx, layer in enumerate(layers):
                if not args.rotate_mode == "none":
                    if not (
                        args.rotate_mode == "quarot" or args.rotate_mode == "spinquant"
                    ):
                        k_rotation = None
                        k_had = True
                    else:
                        k_rotation = U_cpk[f"layer.{idx}.self_attn.key_pos"].cuda()

                        rot2 = random_orthogonal_matrix(residual_length_k, "cuda")
                        rot1 = random_orthogonal_matrix(
                            layer.self_attn.head_dim - residual_length_k, "cuda"
                        )

                        k_rotation_1 = torch.matmul(
                            k_rotation[:, :-residual_length_k], rot1
                        )
                        k_rotation_2 = torch.matmul(
                            k_rotation[:, -residual_length_k:], rot2
                        )
                        k_rotation = torch.cat([k_rotation_1, k_rotation_2], dim=-1)

                        quantizer = quant_utils.WeightQuantizer()
                        quantizer.configure(8)
                        quantizer.find_params(k_rotation)
                        k_rotation = quantizer.quantize(k_rotation)
                        k_had = False

                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                        layer.self_attn,
                        rope_function_name,
                        config=model.config,
                        k_rotation=k_rotation,
                        k_had=k_had,
                        **k_quant_config,
                    )

    return model
