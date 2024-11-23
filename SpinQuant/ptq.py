# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from logging import Logger

import torch
import torch.distributed as dist
from transformers import LlamaTokenizerFast, AutoConfig

from eval_utils.main import ptq_model
from eval_utils.modeling_llama_2 import LlamaForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from utils.LMClass import LMClass
from lm_eval import evaluator
from utils.categories import subcategories, categories
import numpy as np
from lm_eval.utils import make_table

log: Logger = utils.get_logger("spinquant")


@torch.no_grad()
def evaluate(lm, args):
    results = {}

    # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
    testloader = data_utils.get_wikitext2(
        seed=args.seed,
        seqlen=2048,
        tokenizer=lm.tokenizer,
        eval_mode=True,
    )

    # dataset_ppl = eval_utils.evaluator_cuda(lm.model, testloader, utils.DEV, args)
    dataset_ppl = eval_utils.evaluator(lm.model, testloader, utils.DEV, args)
    log.info("wiki2 ppl is: {}".format(dataset_ppl))
    # results["wikitext_ppl"] = dataset_ppl

    if args.tasks != "":
        # bring model to GPU first
        lm.model.to(utils.DEV)
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks.split(","),
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
            batch_size=args.bsz,
        )
        results.update(t_results)
        print(make_table(t_results))
        # print(t_results["results"])
        # for test of MMLU
        if "hendrycksTest" in args.tasks:
            all_cors = []
            all_cors_norm = []
            subcat_cors = {
                subcat: []
                for subcat_lists in subcategories.values()
                for subcat in subcat_lists
            }
            cat_cors = {cat: [] for cat in categories}
            cat_cors_norm = {cat: [] for cat in categories}
            for key in t_results["results"].keys():
                if not "hendrycksTest" in key:
                    continue
                subject = key.split("-")[-1]
                cors = t_results["results"][key]["acc"]
                cors_norm = t_results["results"][key]["acc_norm"]
                subcats = subcategories[subject]
                for subcat in subcats:
                    subcat_cors[subcat].append(cors)
                    for key in categories.keys():
                        if subcat in categories[key]:
                            cat_cors[key].append(cors)
                            cat_cors_norm[key].append(cors_norm)
                    all_cors.append(cors)
                    all_cors_norm.append(cors_norm)

            for cat in cat_cors:
                cat_acc = np.mean(cat_cors[cat])
                log.info("Average accuracy {:.4f} - {}".format(cat_acc, cat))
            weighted_acc = np.mean(all_cors)
            log.info("Average accuracy: {:.4f}".format(weighted_acc))
    return results


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = utils.get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()

    config = AutoConfig.from_pretrained(
        model_args.input_model,
    )
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        torch_dtype=dtype,
        config=config,
    )

    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    for name, m in model.named_modules():
        if "basis_change" in name:
            m.weight.data.copy_(torch.eye(model.config.hidden_size))
    model = ptq_model(ptq_args, model, model_args)
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model.to(utils.DEV))

    lm.model.seqlen = training_args.model_max_length

    results = evaluate(lm, ptq_args)
    # if local_rank == 0:
    #     log.info("Model PTQ completed {}".format(model))
    #     log.info("Start to load tokenizer...")
    # tokenizer = LlamaTokenizerFast.from_pretrained(
    #     pretrained_model_name_or_path=model_args.input_model,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=True,
    #     add_eos_token=False,
    #     add_bos_token=False,
    # )
    # log.info("Complete tokenizer loading...")
    # model.config.use_cache = False

    # testloader = data_utils.get_wikitext2(
    #     seed=ptq_args.seed,
    #     seqlen=2048,
    #     tokenizer=tokenizer,
    #     eval_mode=True,
    # )

    # dataset_ppl = eval_utils.evaluator_cuda(model, testloader, utils.DEV, ptq_args)
    # log.info("wiki2 ppl is: {}".format(dataset_ppl))
    dist.barrier()


if __name__ == "__main__":
    train()
