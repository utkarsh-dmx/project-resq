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
    for n, p in model.named_parameters():
        if "rot_1" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameters :", trainable_params)


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
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        save_strategy="no",
        logging_steps=10,
        num_train_epochs=1,
        max_steps=50,
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
    )
    start_time = perf_counter()
    trainer.train()
    end_time = perf_counter()
    training_time = end_time - start_time
    print(f"Time taken for training: {training_time} seconds")
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = use_cache
    set_model_to_float16(model)


def set_model_to_float16(model):
    model.to(torch.float16)
    for name, param in model.named_parameters():
        if "rot" in name:
            param.data.copy_(param.to(torch.float16))
    # model.to(torch.float16)
