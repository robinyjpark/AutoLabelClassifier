from accelerate import Accelerator
from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModel

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

    path_base_model = cfg.base_model

    logging.info("Loading model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        path_base_model,  
        device_map="auto"
    )
    base_model.config.use_cache = True

    con_model = PeftModel.from_pretrained(base_model, cfg.ntp_model)
    con_model.config.use_cache = True

    logging.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        path_base_model, add_bos_token=True, trust_remote_code=True,
        # padding=True, padding_side="left", 
        truncation=True,
        max_length=1024)

    # tokenizer.pad_token = "[PAD]"

    logging.info("Loading data...")

    if cfg.condition == 'cancer':
        condition = 'cancer'
        data = cfg.labeled_data
        definition = cfg.cancer_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['label']
        column = 'report_no_hist'

    elif cfg.condition == 'cancer_TRAIN':
        condition = 'cancer'
        data = cfg.train_labeled_data
        definition = cfg.cancer_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['label']
        column = 'report_no_hist'

    elif cfg.condition == 'stenosis_TRAIN': 
        condition = 'stenosis'
        data = cfg.osc_train_data
        definition = cfg.stenosis_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['result']
        column = 'report'

    elif cfg.condition == 'stenosis': 
        condition = 'stenosis'
        data = cfg.osc_test_data
        definition = cfg.stenosis_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['result']
        column = 'report'

    elif cfg.condition == 'cauda_equina_train': 
        condition = 'cauda equina'
        data = cfg.ce_train_data
        definition = cfg.cauda_equina_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['global_label']
        column = 'report'
    
    elif cfg.condition == 'cauda_equina_test': 
        condition = 'cauda equina'
        data = cfg.ce_test_data
        definition = cfg.cauda_equina_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['global_label']
        column = 'report'

    elif cfg.condition == 'herniation_train': 
        condition = 'herniation'
        data = cfg.hern_train_data
        definition = cfg.herniation_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['label']
        column = 'report'
    
    elif cfg.condition == 'herniation_test': 
        condition = 'herniation'
        data = cfg.hern_test_data
        definition = cfg.herniation_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_test['label']
        column = 'report'

    elif cfg.condition == 'ALL_STENOSIS': 
        condition = 'stenosis'
        data = '/work/robinpark/PID010A_clean/all_osclmric_reports.csv'
        definition = cfg.stenosis_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        column = 'report'

    elif cfg.condition == 'ALL_CANCER': 
        condition = 'cancer'
        data = '/work/robinpark/NCIMI_clean/segmented_unique_reports.csv'
        definition = cfg.cancer_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        df_test = df_test.drop([1795, 2210]).reset_index(drop=True) # 1795, 2210
        column = 'report_no_hist'

    elif cfg.condition == 'ALL_CAUDA_EQUINA': 
        condition = 'cauda equina'
        data = '/work/robinpark/PID010A_clean/all_osclmric_reports.csv'
        definition = cfg.cauda_equina_definition
        df_test = pd.read_csv(data, low_memory=False, index_col=0)
        column = 'report'

    prompt1 = f"""\
        You are a radiologist. Your job is to diagnose {condition} using a medical report. 
        Tell the truth and answer as precisely as possible.  
    """

    # Print IVD level (if specified)
    print(cfg.ivd)

    # Set tokens for yes and no and specify answer marker
    if cfg.model_name == 'zephyr':
        yes_token = 5081
        no_token = 708
        con_splitter = '<|assistant|>\n'
    elif cfg.model_name == 'llama3':
        yes_token = 9891
        no_token = 2201
        con_splitter = '<|start_header_id|>assistant<|end_header_id|>\n'
    
    answer_marker = con_splitter + 'ANSWER: '

    logging.info("Making prompts...")
    li_results = []
    li_prompts = []
    for i in range(len(df_test)):
        example = df_test[column].iloc[i]

        messages = [
            {"role": "system", "content": f"{prompt1}/nReport: {example}"},
            {"role": "user", "content": f"Can you write a summary for the report with the goal of diagnosing {condition}?"},
        ]
        eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        li_prompts.append(eval_prompt)

    logging.info("Starting inference...")
    con_model.eval()
    li_con = []
    li_results = []
    with torch.no_grad():
        for i in range(len(li_prompts)):
            logging.info(f"Executing {i+1} of {len(li_prompts)}...")
            input_ids = tokenizer(li_prompts[i], return_tensors="pt")['input_ids'].to('cuda')
            gen_ids = con_model.generate(input_ids=input_ids, max_new_tokens=300, repetition_penalty=1.15,do_sample=False)[0] ## MAX NEW TOKENS CHANGED
            output = tokenizer.decode(gen_ids, skip_special_tokens=False)
            answer = output.split(con_splitter)[-1]
            if cfg.model_name=='llama3':
                answer = answer.replace('<|eot_id|>','')
            li_con.append(answer)
            li_results.append(output)

    if cfg.ivd:
        ivd = f" at {cfg.ivd_level}"
        ivd_level = cfg.ivd_level.replace('-','')
    else:
        ivd=''
        ivd_level=''
    
    print('ivd_level: ', ivd_level)

    li_results2 = []
    for i in range(len(df_test)):
        example = df_test[column].iloc[i]

        messages = [
            {"role": "system","content": f"{prompt1}/nReport: {example}"},
            {"role": "user", "content": f"Can you write a summary for the report with the goal of diagnosing {condition}?"},
            {"role": "assistant", "content": f"{li_con[i]}"},
            {"role": "user", "content": f"{definition} Based on your summary, does the patient have {condition}{ivd}? Answer 'yes' for yes, 'no' for no. Only output one token after 'ANSWER: '"} 
        ]
        eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        li_results2.append(eval_prompt + answer_marker) 
    print(li_results2[0])

    li_yes = []
    li_no = []
    base_model.eval()
    with torch.no_grad():
        for i in range(len(li_results2)):
            logging.info(f"Executing {i+1} of {len(li_results2)}...")
            input_ids = tokenizer(li_results2[i], return_tensors="pt")['input_ids'].to('cuda')
            model_out = base_model(input_ids)
            final_output_token_logits = model_out.logits[0][-1]
            yes_score = final_output_token_logits[yes_token].cpu().item() 
            no_score = final_output_token_logits[no_token].cpu().item() 
            li_yes.append(yes_score)
            li_no.append(no_score)
    
    labeled_reports = pd.DataFrame(
        {'report_no_hist': df_test[column],
        'pred_conclusion': li_con,
        'labels': labels,
        'yes_score': li_yes,
        'no_score': li_no})
    
    labeled_reports['results'] = 0
    labeled_reports.loc[
        labeled_reports.yes_score > labeled_reports.no_score, 
        'results'] = 1
    
    labeled_reports.to_csv(f'{cfg.llm_results_path}/april2024/summary-query/sft-base/{cfg.model_name}_con_lora_base_2step_{cfg.condition}{ivd_level}_new_template_yesno_scores_have_spec_summary_prompt.csv')

if __name__ == "__main__":
    run_llm()