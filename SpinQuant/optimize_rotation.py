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
from transformers import (
    LlamaTokenizerFast,
    Trainer,
    default_data_collator,
    TrainingArguments,
)
from utils.data_utils import get_wikitext2
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant_2 import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from train_utils.optimizer import SGDG
from utils.data_utils import CustomJsonDataset
from utils.hadamard_utils import (
    random_hadamard_matrix,
    get_hadK,
    random_orthogonal_matrix,
)
import math
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict

# import eval_utils, utils.utils, utils.data_utils
from utils import data_utils, eval_utils, utils

log: Logger = get_logger("spinquant")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    perplexity = torch.exp(loss)
    return {"perplexity": perplexity.item()}


class RotateModule(nn.Module):
    def __init__(self, R_init):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.to(torch.float32).to(torch.device("cuda")))

    def forward(self, x, transpose=False):
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=dtype,
    )
    breakpoint()

    model = prepare_model(ptq_args, model)
    model_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    mlp_dim = model.config.intermediate_size
    head_dim = model_dim // num_heads
    residual_length = int(ptq_args.residual_fraction * model_dim)
    print(residual_length)
    for param in model.parameters():
        param.requires_grad = False
    if ptq_args.residual_fraction > 0:
        R1_1 = random_orthogonal_matrix(residual_length, "cuda")
    R1_2 = random_orthogonal_matrix(model_dim - residual_length, "cuda")
    model.model.R1_1 = RotateModule(R1_1)
    model.model.R1_2 = RotateModule(R1_2)

    residual_length = int(ptq_args.residual_fraction * head_dim)
    print(residual_length)

    if ptq_args.residual_fraction > 0:
        R2_1 = random_orthogonal_matrix(residual_length, "cuda")

    R2_2 = random_orthogonal_matrix(
        head_dim - residual_length,
        "cuda",
    )

    residual_length = int(ptq_args.residual_fraction * ptq_args.down_proj_blocksize)
    if ptq_args.residual_fraction > 0:
        R4_1 = random_orthogonal_matrix(residual_length, "cuda")

    R4_2 = random_orthogonal_matrix(
        ptq_args.down_proj_blocksize - residual_length,
        "cuda",
    )
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model

        model.model.layers[i].R1_attn_1 = RotateModule(R1_1)
        model.model.layers[i].R1_attn_2 = RotateModule(R1_2)
        model.model.layers[i].R1_mlp_1 = RotateModule(R1_1)
        model.model.layers[i].R1_mlp_2 = RotateModule(R1_2)

        model.model.layers[i].self_attn.R2_1 = RotateModule(R2_1)
        model.model.layers[i].self_attn.R2_2 = RotateModule(R2_2)

        # R4, _ = get_hadK(model.config.intermediate_size)
        # model.model.layers[i].mlp.R4 = RotateModule(R4)
        model.model.layers[i].mlp.R4_1 = RotateModule(R4_1)
        model.model.layers[i].mlp.R4_2 = RotateModule(R4_2)
    if local_rank == 0:
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

    #####
    model.seqlen = training_args.model_max_length
    testloader = data_utils.get_wikitext2(
        seed=ptq_args.seed,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )

    # dataset_ppl = eval_utils.evaluator_cuda(model, testloader, utils.DEV, ptq_args)
    # log.info("wiki2 ppl is: {}".format(dataset_ppl))
    testset = testloader.input_ids
    nsamples = testset.numel() // model.seqlen
    testset = (
        testset[:, : nsamples * model.seqlen].view(nsamples, model.seqlen).to(utils.DEV)
    )  # (nsamples, seqlen)
    batch_size = ptq_args.bsz
    testset = [testset[i : i + batch_size] for i in range(0, nsamples, batch_size)]

    #####
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )

    test_data = CustomJsonDataset(
        calibration_datasets["test"], tokenizer, block_size=model.seqlen
    )

    # trainable_parameters = [model.R1.weight] + [
    #     model.model.layers[i].self_attn.R2.weight
    #     for i in range(model.config.num_hidden_layers)
    # ]

    trainable_parameters = []
    # trainable_parameters.append(
    #     {
    #         "params": model.R1_1.weight,
    #         "stiefel": True,
    #         "lr": training_args.learning_rate,
    #         "momentum": 0.9,
    #         "nesterov": True,
    #     }
    # )
    # trainable_parameters.append(
    #     {
    #         "params": model.R1_2.weight,
    #         "stiefel": True,
    #         "lr": training_args.learning_rate,
    #         "momentum": 0.9,
    #         "nesterov": True,
    #     }
    # )
    for i in range(model.config.num_hidden_layers):
        trainable_parameters.append(
            {
                "params": model.model.layers[i].self_attn.R2_1.weight,
                "stiefel": False,
                "lr": training_args.learning_rate,
                "momentum": 0.9,
                "nesterov": True,
            }
        )

        trainable_parameters.append(
            {
                "params": model.model.layers[i].self_attn.R2_2.weight,
                "stiefel": False,
                "lr": training_args.learning_rate,
                "momentum": 0.9,
                "nesterov": True,
            }
        )

        trainable_parameters.append(
            {
                "params": model.model.layers[i].mlp.R4_2.weight,
                "stiefel": False,
                "lr": training_args.learning_rate,
                "momentum": 0.9,
                "nesterov": True,
            }
        )
    for name, p in model.named_parameters():
        if (
            "R1_1.weight" in name
            or "R1_2.weight" in name
            or "R2_1.weight" in name
            or "R2_2.weight" in name
            or "R4.weight" in name
        ):
            p.requires_grad = True

    model.seqlen = training_args.model_max_length
    # optimizer = SGDG(trainable_parameters, lr=training_args.learning_rate, stiefel=True)
    # optimizer = SGDG(
    #     trainable_parameters, lr=training_args.learning_rate, stiefel=True, momentum=0.9
    # )
    optimizer = SGDG(
        trainable_parameters,
        lr=training_args.learning_rate,
    )
    MyTrainer = Trainer
    # Use FSDP for 70B rotation training
    if training_args.fsdp != "" and training_args.fsdp != []:
        MyTrainer = FSDPTrainer

    training_args = TrainingArguments(
        output_dir="./checkpoint",
        save_strategy="steps",
        save_steps=0.1,
        resume_from_checkpoint=True,
        save_safetensors=False,
        logging_steps=0.1,
        gradient_checkpointing=True,
        # load_best_model_at_end=True,  # Load the best model at the end of training
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        max_steps=training_args.max_steps,
        # logging_first_step=True,
        # metric_for_best_model="perplexity",
        # greater_is_better=False,
        # eval_strategy="steps",
        # evaluation_strategy="steps",
        # eval_steps=0.1,
        auto_find_batch_size=True,
        # eval_accumulation_steps=5,
        save_total_limit=3,
    )
    # training_args.output_dir = "./checkpoint"
    # training_args.save_strategy = "steps"
    # training_args.save_steps = 0.1
    # training_args.resume_from_checkpoint = True
    # training_args.save_safetensors = False
    # training_args.logging_steps = 0.1
    # training_args.gradient_checkpointing = True
    # training_args.load_best_model_at_end = (
    # True,
    # )  # Load the best model at the end of training
    # training_args.metric_for_best_model = (
    # "perplexity",
    # )  # Use perplexity to select the best model
    # training_args.greater_is_better = (False,)  # Lower perplexity is better
    # training_args.eval_strategy = ("steps",)
    # training_args.eval_steps = 1

    # training_args.optim = "adamw_torch"
    model.gradient_checkpointing_enable()
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics,  # Make sure this is included
    )
    torch.distributed.barrier()
    # breakpoint()

    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    torch.cuda.empty_cache()
    torch.distributed.barrier()
    # trainer.train(resume_from_checkpoint=True)
    # trainer.train()

    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()

    R_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if "R1_1.weight" in key
        or "R1_2.weight" in key
        or "R1_attn_1" in key
        or "R1_attn_2" in key
        or "R1_mlp_1" in key
        or "R1_mlp_2" in key
        or "self_attn.R2_1" in key
        or "self_attn.R2_2" in key
        or "mlp.R4" in key
        or "mlp.R4_1" in key
        or "mlp.R4_2" in key
    }
    # print(R_dict.keys())
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(
            model_args.output_rotation_path,
            "R-" + model_args.input_model.split("/")[1] + ".bin",
        )
        torch.save(
            R_dict,
            path,
        )
        # also remove the checkpoint folder, we dont need it not
        # os.rmdir(training_args.output_dir)
    dist.barrier()


if __name__ == "__main__":
    train()
