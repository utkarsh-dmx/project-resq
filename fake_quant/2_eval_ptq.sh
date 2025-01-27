# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### k_groupsize, v_groupsize = 64 only for Llama-3.2-1B else 128
#### Storing config for 70B model
torchrun --nnodes=1 --nproc_per_node=1 --master_port=24550 ptq.py \
--input_model Qwen/Qwen2.5-72B \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits 4 \
--a_bits 4 \
--k_bits 4 \
--v_bits 4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--residual_fraction 0.125 \
--rotate_mode "none" \
--optimized_rotation_path ./rotation/R-high_prec-0.125-sparse-0.0-Qwen2.5-32B.bin \
--optimized_basis_path ./rotation/U-wikitext-512-Qwen2.5-32B.bin \
--rotation_granularity 'full_shared' \
--tasks "mmlu,boolq,piqa,social_iqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa" \
--rotate \
--multigpu \