# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### k_groupsize, v_groupsize = 64 only for Llama-3.2-1B else 128
torchrun --nnodes=1 --nproc_per_node=1 collect_activations.py \
--input_model meta-llama/Meta-Llama-3-8B \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 8 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 16 \
--a_bits 16 \
--k_bits 16 \
--v_bits 16 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--residual_fraction 0.125 \
--optimized_basis_path ./rotation/U-full_sharedMeta-Llama-3-8B.bin \
--optimized_rotation_path ./rotation/R-Meta-Llama-3-8B.bin \
--output_dir "output/" \
--rotate_mode 'none' \
--rotation_granularity 'full_shared' \
--capture_layer_io \
--layer_idx 14 \
