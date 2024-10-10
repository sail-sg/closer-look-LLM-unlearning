import os
import shutil
import warnings
from pathlib import Path

import datasets
import hydra
import torch
import transformers
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from dataset import TextForgetDatasetQA, custom_data_collator_forget
from trainer import CustomTrainerForgetting
from utils import get_model_identifiers_from_yaml, set_random_seed

warnings.filterwarnings('ignore')


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


@hydra.main(version_base=None, config_path="config", config_name="real_world")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    seed = cfg.seed
    set_random_seed(seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    config = AutoConfig.from_pretrained(model_id)

    curr_save_dir = cfg.save_dir

    if os.path.exists(os.path.join(curr_save_dir, 'eval_results-last', 'unlearning_results.txt')):
        print(f'Task already unlearned.')
        exit()

    if local_rank == 0:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    forget_data = datasets.load_dataset('json', data_files=os.path.join(cfg.data_path, cfg.split + '.json'),
                                        split='train')
    retain_data = datasets.load_dataset('json', data_files=os.path.join(cfg.data_path, cfg.retain + '.json'),
                                        split='train')

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    torch_format_dataset = TextForgetDatasetQA(tokenizer=tokenizer,
                                               model_family=cfg.model_family,
                                               forget_data=forget_data,
                                               retain_data=retain_data,
                                               max_length=300,
                                               mask=cfg.mask)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(
        torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
                batch_size * gradient_accumulation_steps * num_devices)
    warmup_steps = steps_per_epoch if steps_per_epoch > 1 else 0

    if cfg.save_steps == 'steps_per_epoch':
        save_steps = steps_per_epoch
    elif cfg.save_steps == 'last':
        save_steps = max_steps
    else:
        save_steps = cfg.save_steps

    if local_rank == 0:
        print("Saving to: ", curr_save_dir)

    # load the config files for deepspeed
    if cfg.use_LoRA:
        ds_config = 'config/ds_config/lora.json'
    else:
        ds_config = 'config/ds_config/llama3.json'

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        bf16_full_eval=True,
        output_dir=curr_save_dir,
        optim="paged_adamw_32bit",
        deepspeed=ds_config,
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        evaluation_strategy="no",
    )

    # load target LLM
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        config=config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
    )
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    # Configure LoRA parameters
    if cfg.use_LoRA:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=find_all_linear_names(model),
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            lora_dropout=cfg.LoRA.dropout,
        )
        model = get_peft_model(model, peft_config)

    # load reference model
    reference_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        config=config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
    )
    reference_model = reference_model.eval()

    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        # the callback for computing metrics, None in this case since you're doing it in your callback
        compute_metrics=None,
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=custom_data_collator_forget,
        loss_type=cfg.forget_loss,
        ref_model=reference_model,
        beta=cfg.beta,
        forget_coeff=cfg.forget_coeff,
        regularization_coeff=cfg.regularization_coeff,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print('Start Training ...')
    # Start training
    trainer.train()

    if local_rank == 0:
        if os.path.exists(os.path.join(curr_save_dir, f'checkpoint-{max_steps}')):
            shutil.move(os.path.join(curr_save_dir, f'checkpoint-{max_steps}'),
                        os.path.join(curr_save_dir, f'checkpoint-last'))


if __name__ == "__main__":
    main()
