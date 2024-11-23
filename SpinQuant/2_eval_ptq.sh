# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

### k_groupsize, v_groupsize = 64 only for Llama-3.2-1B else 128
torchrun --nnodes=1 --nproc_per_node=1 --master_port=24559 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 8 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $5 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--rotate \
--residual_fraction $6 \
--optimized_rotation_path $7 \
--optimized_basis_path $8 \
--rotation_granularity $9 \
--tasks "mmlu" \
--load_qmodel_path ./resq-meta-llama-3-8b \
