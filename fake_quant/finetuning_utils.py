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
from trl import SFTTrainer


def set_trainable_params(model):
    for n, p in model.named_parameters():
        if "rot" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameters :", trainable_params)


def do_finetuning(model, args):
    # get alpaca dataset for finetuning
    ds = load_dataset("tatsu-lab/alpaca")
    ds = ds.remove_columns(["input", "output", "instruction"])
    traindata = ds["train"]

    model.config.use_cache = False
    set_trainable_params(model)

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
    training_arguments = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=10,
        fp16=True,
        push_to_hub=False,
        gradient_checkpointing=True,
    )

    # creating trainer with the training agruments
    model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        train_dataset=traindata,
        dataset_text_field="text",  # mentioned the required column
        args=training_arguments,  # training agruments
        tokenizer=tokenizer,  # tokenizer
        packing=False,
        max_seq_length=2048,
    )

    start_time = perf_counter()
    trainer.train()
    end_time = perf_counter()
    training_time = end_time - start_time
    print(f"Time taken for training: {training_time} seconds")
