import os
import json
import time
import hydra
import openai
import logging
import pandas as pd

from dotenv import load_dotenv
from omegaconf import DictConfig

@hydra.main(version_base="1.2", config_path="../../../conf", config_name="config")
def save_api_results(cfg: DictConfig) -> None:

    # API key
    load_dotenv()
    
    client = openai.OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
    )

    if cfg.condition == 'cancer_TRAIN':
        condition = 'cancer'
        data = cfg.train_labeled_data
        definition = cfg.cancer_definition
        df_labeled = pd.read_csv(data, low_memory=False, index_col=0).reset_index(drop=True)
        labels = df_labeled['label']
        column = 'report_no_hist'

    elif cfg.condition == 'stenosis_TRAIN': 
        condition = 'stenosis'
        data = cfg.osc_train_data
        definition = cfg.stenosis_definition
        df_labeled = pd.read_csv(data, low_memory=False, index_col=0).reset_index(drop=True)
        labels = df_labeled['result']
        column = 'report'

    elif cfg.condition == 'cauda_equina_test': 
        condition = 'cauda equina'
        data = cfg.ce_test_data
        definition = cfg.cauda_equina_definition
        df_labeled = pd.read_csv(data, low_memory=False, index_col=0).reset_index(drop=True)
        labels = df_labeled['global_label']
        column = 'report'

    elif cfg.condition == 'herniation_test': 
        condition = 'herniation'
        data = cfg.hern_test_data
        definition = cfg.herniation_definition
        df_labeled = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_labeled['global_label']
        column = 'report'
    
    elif cfg.condition == 'spon_test': 
        condition = 'spondylolisthesis'
        data = cfg.spon_test_data
        definition = cfg.spon_definition
        df_labeled = pd.read_csv(data, low_memory=False, index_col=0)
        labels = df_labeled['global_label']
        column = 'report'

    user_prompt = f"{definition} Based on the report, does the patient have {condition}? Answer 'yes' for yes, 'no' for no. Only output one token after 'ANSWER: '."

    li_executed_prompts = []
    li_labels = []
    li_gpt_response = []
    for i in range(len(df_labeled)):
        system_prompt = f"""\
        You are a radiologist. Your job is to diagnose {condition} using a medical report. 
        Tell the truth and answer as precisely as possible.  

        Report: {df_labeled[column][i]}
        """

        if i % 10 == 0:
            print(f"{i} of {len(df_labeled)} reports executing...")
            print(system_prompt)

        response = client.chat.completions.create(
            model=cfg.gpt_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        li_executed_prompts.append(df_labeled[column].iloc[i])
        li_labels.append(labels.iloc[i])
        response_message = response.choices[0].message.content
        li_gpt_response.append(response_message)

        # Write out every loop to avoid losing output
        gpt_reports = pd.DataFrame(
            {
                'report_no_hist': li_executed_prompts,
                'labels': li_labels,
                'results': li_gpt_response
            }
        )
        
        gpt_reports.to_csv(f'{cfg.llm_results_path}/april2024/direct-query/gpt4_{cfg.condition}_new_have_prompt.csv')
        time.sleep(20)

if __name__ == "__main__":
    save_api_results()