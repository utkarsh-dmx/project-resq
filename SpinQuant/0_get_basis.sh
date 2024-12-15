# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python get_basis.py \
--input_model meta-llama/Llama-3.2-3B  \
--output_rotation_path "rotation" \
--model_max_length 2048 \
--down_proj_blocksize 256 \
--residual_fraction 0.125 \
--rotation_granularity "full_shared" \
--rotate_mode "resq" \
--nsamples 512 \
--calib_dataset "wikitext" \