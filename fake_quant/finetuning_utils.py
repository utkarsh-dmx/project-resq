from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model
from time import perf_counter
import transformers
from trl import SFTTrainer, SFTConfig
from cayley_utils import SGDG, AdamG, CayleyAdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import data_utils
from accelerate import Accelerator, load_checkpoint_and_dispatch
from torch.optim import AdamW, SGD
from transformers import get_scheduler
from tqdm.auto import tqdm
from eval_utils import evaluator_cuda
import os
import logging

from torch.utils.data import DataLoader

# accelerate launch  main.py --model meta-llama/Llama-2-7b-hf  --rotate --a_bits 4 --v_bits 4 --k_bits 4 --w_bits 4 --w_clip --rotate_mode learnable --seed 123 --save_name temp


class CustomTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer after adding orthogonality loss. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Orthogonality loss
        # loss_2 = 0.0
        # for n, p in model.named_parameters():
        #     if "rot_1" in n:
        #         ones = torch.ones(p.shape[0], device=p.device)
        #         # loss_2 += torch.dist(p @ p.t(), torch.eye(p.shape[0]).to(p.device))
        #         loss_2 += torch.norm((p @ p.t()) - torch.diag(ones), "fro")
        # loss = loss + 0.001 * loss_2
        return (loss, outputs) if return_outputs else loss


def set_trainable_params(model):
    params = []
    for n, p in model.named_parameters():
        # if "rot_1" in n or "rot_2" in n:
        if "rot_1" in n:
            p.requires_grad = True
            params.append(p)
        else:
            p.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable Parameters :{trainable_params}")
    # print("Trainable Parameters :", trainable_params)
    # logging.info("Trainable Parameters :", trainable_params)
    return params


def do_finetuning(model, args):
    # get alpaca dataset for finetuning
    # ds = load_dataset("tatsu-lab/alpaca")
    # ds = ds.remove_columns(["input", "output", "instruction"])
    # traindata = ds["train"]
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    evaldata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # model.config.use_cache = False
    use_cache = model.config.use_cache
    model.config.use_cache = False
    params = set_trainable_params(model)
    initial_rot_1 = model.model.rot_1.clone().detach()
    initial_rot_2 = model.model.layers[0].self_attn.rot_2.clone().detach()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        legacy=False,
        trust_remote_code=True,
        return_token_type_ids=False,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    output_dir = (
        "/home/coder/codebase/Learnable-Rotations-LLM-Quant/fake_quant/finetune_logs/"
    )
    # lr = 0.0
    # optim = SGDG(params, lr, stiefel=True, nesterov=True, momentum=0.9)
    # optim = AdamG(params, lr, grassmann=True, nesterov=True, momentum=0.9)
    # optim = CayleyAdamW(params, lr, grassmann=True, nesterov=True, momentum=0.9)
    # lr_scheduler = CosineAnnealingLR(optim, T_max=50)
    # optimizers = (optim, lr_scheduler)
    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        save_strategy="no",
        logging_steps=1,
        num_train_epochs=1,
        max_steps=5,
        fp16=True,
        push_to_hub=False,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
    )
    # creating trainer with the training agruments
    # model.gradient_checkpointing_enable()
    trainer = CustomTrainer(
        model=model,
        train_dataset=traindata,
        dataset_text_field="text",  # mentioned the required column
        args=training_arguments,  # training agruments
        tokenizer=tokenizer,  # tokenizer
        packing=False,
        max_seq_length=2048,
        eval_dataset=evaldata,
        # optimizers=optimizers,
    )
    start_time = perf_counter()
    trainer.train()
    end_time = perf_counter()
    training_time = end_time - start_time
    print(f"Time taken for training: {training_time} seconds")
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = use_cache
    final_rot_2 = model.model.layers[0].self_attn.rot_2
    final_rot_1 = model.model.rot_1
    print(torch.dist(initial_rot_1.to(final_rot_1.device), final_rot_1))
    print(torch.dist(initial_rot_2.to(final_rot_2.device), final_rot_2))
    breakpoint()
    set_model_to_float16(model)


def do_finetuning_v2(model, args):
    testloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=True,
        batch_size=args.bsz,
    )
    testloader = DataLoader(testloader, batch_size=args.bsz)

    trainloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=False,
        nsamples=1400,
    )
    trainloader = DataLoader(trainloader)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    seqlen = model.seqlen
    params = set_trainable_params(model)

    output_dir = os.path.join(args.save_path, "model")

    accelerator = Accelerator(mixed_precision="bf16")
    # accelerator = Accelerator()
    optimizer = AdamW(params, lr=2e-4, weight_decay=0.0)
    # optimizer = SGD(params, lr=0.01, momentum=0.1, nesterov=True, weight_decay=0.0)

    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    model, optimizer, trainloader, testloader = accelerator.prepare(
        model, optimizer, trainloader, testloader
    )
    if accelerator.is_main_process:
        print(model)
    num_train_epochs = 0
    num_update_steps_per_epoch = len(trainloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    gradient_accumulation_steps = 16
    logging_steps = 5
    eval_steps = 5
    best_ppl = 1e10  # set to very high value.
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=0,
    )
    completed_steps = 0
    start_time = perf_counter()
    accum_loss = 0.0
    for epoch in range(num_train_epochs):
        for step, batch in tqdm(
            enumerate(trainloader, start=1),
            total=num_training_steps,
        ):
            loss = model(input_ids=batch[0], labels=batch[1]).loss
            loss = loss / gradient_accumulation_steps
            accum_loss += loss
            if step % (logging_steps * gradient_accumulation_steps) == 0:
                if accelerator.is_main_process:
                    logging.info(
                        f"samples:{step}, steps: {completed_steps}, loss/train: {accum_loss.item()}"
                    )
                accum_loss = 0.0
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
            if (step % (eval_steps * gradient_accumulation_steps) == 0) or (
                step == num_training_steps
            ):
                model.eval()
                nlls = []
                with torch.no_grad():
                    for batch in testloader:
                        input_ids = batch
                        labels = input_ids.clone()
                        labels[:, :-1] = -100

                        neg_log_likelihood = model(
                            input_ids=input_ids, labels=labels
                        ).loss
                        neg_log_likelihood = accelerator.gather_for_metrics(
                            neg_log_likelihood
                        )
                        nlls.append(neg_log_likelihood)

                    nlls_tensor = torch.cat(nlls)
                    eval_ppl = torch.exp(nlls_tensor.mean())
                if eval_ppl < best_ppl:
                    best_ppl = eval_ppl
                    unwrap_model = accelerator.unwrap_model(model)
                    accelerator.save_model(unwrap_model, output_dir)
                if accelerator.is_main_process:
                    logging.info(f"ppl/eval: {eval_ppl}, ppl/eval_best: {best_ppl}")
                model.train()
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    load_checkpoint_and_dispatch(model, output_dir)
    with torch.no_grad():
        nlls = []
        for batch in testloader:
            input_ids = batch
            labels = input_ids.clone()
            labels[:, :-1] = -100
            neg_ll = model(input_ids=input_ids, labels=labels).loss
            neg_ll = accelerator.gather(neg_ll)
            nlls.append(neg_ll)
        nlls_tensor = torch.stack(nlls)
        eval_ppl = torch.exp(nlls_tensor.mean())
    print(eval_ppl)
    # model.load_state_dict(torch.load(, map_location="cpu"))
    breakpoint()
    end_time = perf_counter()
    training_time = end_time - start_time
    # print(f"Time taken for training: {training_time} seconds")
    logging.info(f"Time taken for training: {training_time} seconds")
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = use_cache
    breakpoint()
    # final_rot_2 = model.model.layers[0].self_attn.rot_2
    # final_rot_1 = model.model.rot_1
    # print(torch.dist(initial_rot_1.to(final_rot_1.device), final_rot_1))
    # print(torch.dist(initial_rot_2.to(final_rot_2.device), final_rot_2))
    set_model_to_float16(model)


def set_model_to_float16(model):
    model.to(torch.float16)
    for name, param in model.named_parameters():
        if "rot" in name:
            param.data.copy_(param.to(torch.float16))


def save_best_model(model, best_ppl, eval_ppl, output_dir):

    return best_ppl
