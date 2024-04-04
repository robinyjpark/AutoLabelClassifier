import os
import torch
import wandb
import hydra
import pandas as pd
import numpy as np

wandb.login()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime 
from omegaconf import DictConfig
from accelerate import Accelerator


@hydra.main(version_base="1.2", config_path="../../../conf", config_name="config")
def train_zephyr(cfg: DictConfig) -> None:
    model_name = "HuggingFaceH4/zephyr-7b-beta"

    wandb_project = "zephyr-finetune"
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map=device_map,
                                                 torch_dtype=torch.float16,
                                                 quantization_config=bnb_config)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    model.config.use_cache = False  # Re-enable for inference!
    model.ddp_find_unused_parameters = False
    model.find_unused_parameters = False

    df_reports = pd.read_csv(cfg.conclusion_training_data)
    train, eval = train_test_split(df_reports[['report_no_hist']], test_size=500)
    train_dataset = Dataset.from_pandas(train.reset_index(drop=True), split='train')
    eval_dataset = Dataset.from_pandas(eval.reset_index(drop=True), split='train')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_eos_token=True,
        add_bos_token=True,
        device_map=device_map
    )
    tokenizer.pad_token = "[PAD]"

    # Step 2: Tokenize the text data
    def generate_and_tokenize_prompt(example):
        prompt = example['report_no_hist']
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=5000,
        ) 
        result["labels"] = [tokenizer.pad_token_id] + result["input_ids"][1:]
        target_word_ids = [28792,3185,3100,2252,1702,28793]
        target_word_index = np.where(np.all(
            np.lib.stride_tricks.sliding_window_view(
                result["labels"], 
                len(target_word_ids)) == target_word_ids, axis=-1))[0][0]
            
        # Mask everything in label before the conclusion
        mask_tokens = [-100] * target_word_index
        result["labels"][:target_word_index] = mask_tokens

        return result

    tokenized_train_data = train_dataset.map(generate_and_tokenize_prompt).remove_columns('report_no_hist')
    tokenized_eval_data = eval_dataset.map(generate_and_tokenize_prompt).remove_columns('report_no_hist')

    # Shuffle
    tokenized_train_data = tokenized_train_data.shuffle(seed=1)
    tokenized_eval_data = tokenized_eval_data.shuffle(seed=1)

    project = "uprep-conclusion-lora"
    base_model_name = "zephyr"
    run_name = base_model_name + "-" + project
    output_dir = os.path.join(cfg.model_weights_save_path, run_name)

    accelerator = Accelerator()

    trainer = accelerator.prepare(Trainer(
        model=model,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_eval_data,
        args=TrainingArguments(
            output_dir=output_dir,
            warmup_steps=20,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            learning_rate=1e-5, 
            optim="paged_adamw_8bit",
            logging_steps=4000,          # When to start reporting loss
            logging_dir="./logs",        # Directory for storing logs
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=4000,             # Save checkpoints every 10 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=4000,             # Evaluate and save checkpoints every 10 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # Name of the W&B run
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':False}
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    ))

    trainer.train()

if __name__ == "__main__":
    train_zephyr()
