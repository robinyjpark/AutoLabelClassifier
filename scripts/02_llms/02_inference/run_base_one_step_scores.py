from accelerate import Accelerator
from datasets import Dataset
from omegaconf import DictConfig

import hydra
import torch
import pandas as pd
import logging


@hydra.main(version_base="1.2", config_path="../../../conf", config_name="config")
def run_llm(cfg: DictConfig) -> None:

    # Transformers cache path must be changed before transformers input
    import os
    os.environ['TRANSFORMERS_CACHE'] = cfg.model_weights_save_path
    print(os.getenv('TRANSFORMERS_CACHE'))

    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.pipelines.pt_utils import KeyDataset

    path_model = "HuggingFaceH4/zephyr-7b-beta"

    logging.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        path_model,  
        device_map="auto"
    )
    model.config.use_cache = True

    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta", add_bos_token=True, trust_remote_code=True,
        padding=True, padding_side="left", truncation=True,
        max_length=1024)

    tokenizer.pad_token = "[PAD]"

    logging.info("Loading data...")

    if cfg.condition == 'cancer':
        condition = 'cancer'
        data = cfg.labeled_data
        definition = cfg.cancer_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['cancer_in_image']
        column = 'report_no_hist'

    elif cfg.condition == 'cancer_TRAIN':
        condition = 'cancer'
        data = cfg.train_labeled_data
        definition = cfg.cancer_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['cancer_in_image']
        column = 'report_no_hist'

    elif cfg.condition == 'stenosis_TRAIN': 
        condition = 'stenosis'
        #data = '/work/robinpark/PID010A_clean/all_osclmric_reports.csv'
        data = cfg.osc_train_data
        definition = cfg.stenosis_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['result']
        column = 'report'

    elif cfg.condition == 'stenosis': 
        condition = 'stenosis'
        #data = '/work/robinpark/PID010A_clean/all_osclmric_reports.csv'
        data = cfg.osc_test_data
        definition = cfg.stenosis_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['result']
        column = 'report'

    prompt1 = f"""\
        You are a radiologist. Your job is to diagnose {condition} using a medical report. 
        Tell the truth and answer as precisely as possible.  
    """

    logging.info("Making prompts...")
    li_prompts = []
    for i in range(len(df_test)):
        example = df_test[column].iloc[i]

        messages = [
            {"role": "system","content": f"{prompt1}/nReport: {example}"},
            {"role": "user", "content": f"{definition} Based on the report, does the patient have {cfg.condition}? Answer 'yes' for yes, 'no' for no. Only output one token after 'ANSWER: "} 
        ]
        eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        li_prompts.append(eval_prompt + '<|assistant|>\nANSWER: ') 

    print(li_prompts[0])

    li_yes = []
    li_no = []
    with torch.no_grad():
        for i in range(len(li_prompts)):
            logging.info(f"Executing {i+1} of {len(li_prompts)}...")
            input_ids = tokenizer(li_prompts[i], return_tensors="pt")['input_ids'].to('cuda')
            model_out = model(input_ids)
            final_output_token_logits = model_out.logits[0][-1]
            yes_score = final_output_token_logits[5081].cpu().item() # %:1239 / yes:5081
            no_score = final_output_token_logits[708].cpu().item() # $:429 / no:708
            li_yes.append(yes_score)
            li_no.append(no_score)
    
    labeled_reports = pd.DataFrame(
        {'report_no_hist': df_test[column],
        'labels': labels,
        'yes_score': li_yes,
        'no_score': li_no})
    
    labeled_reports['results'] = 0
    labeled_reports.loc[
        labeled_reports.yes_score > labeled_reports.no_score, 
        'results'] = 1
    
    labeled_reports.to_csv(f'{cfg.clean_path}/base_one_step_scores/base_1step_{cfg.condition}_new_template_yesno_scores_have_prompt.csv')

if __name__ == "__main__":
    run_llm()