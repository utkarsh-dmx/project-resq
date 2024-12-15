# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
### k_groupsize, v_groupsize = 64 only for Llama-3.2-1B else 128

torchrun --nnodes=1 --nproc_per_node=1 optimize_rotation.py \
--input_model meta-llama/Meta-Llama-3-3B  \
--output_rotation_path "rotation" \
--output_dir "output/" \
--logging_dir "logs/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 8 \
--logging_steps 1 \
--learning_rate 2.0 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--max_steps 150 \
--w_bits 16 \
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
--down_proj_blocksize 4096 \
--optimized_basis_path ./rotation/U-Llama-3.2-3B.bin \


