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
from accelerate import Accelerator
from torch.optim import AdamW, SGD
from transformers import get_scheduler
from tqdm import tqdm
from eval_utils import evaluator_cuda


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
    print("Trainable Parameters :", trainable_params)
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
    )
    trainloader = data_utils.get_loaders(
        args.eval_dataset,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        hf_token=args.hf_token,
        eval_mode=False,
        nsamples=400,
    )
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

    accelerator = Accelerator(mixed_precision="fp16")
    # optimizer = AdamW(params, lr=2e-4, weight_decay=0.0)
    optimizer = SGD(params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0)
    model, optimizer, trainloader, testloader = accelerator.prepare(
        model, optimizer, trainloader, testloader
    )
    num_train_epochs = 1
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
    model.train()
    model.gradient_checkpointing_enable()
    completed_steps = 0
    start_time = perf_counter()
    for epoch in range(nu