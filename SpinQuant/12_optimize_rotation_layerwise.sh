# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python get_basis.py \
--input_model meta-llama/Llama-2-7b-hf  \
--output_rotation_path "rotation" \
--output_dir "output/" \
--logging_dir "logs/" \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--per_device_train_batch_size 32 \
--logging_steps 1 \
--learning_rate 0.0001 \
--weight_decay 0. \
--lr_scheduler_type "cosine" \
--gradient_checkpointing True \
--max_steps 100 \
--w_bits 16 \
--a_bits 3 \
--k_bits 3 \
--v_bits 3 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
