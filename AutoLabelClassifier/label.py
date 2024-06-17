"""
Code for labelling the rpeorts
"""

import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.pipelines.pt_utils import KeyDataset


@torch.no_grad()
def generate_summary(model, tokenizer, prompt, device):
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    gen_ids = model.generate(
        input_ids=input_ids, max_new_tokens=1024, repetition_penalty=1.15
    )[0]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    answer = output.split("<|assistant|>")[-1]
    return answer


@torch.no_grad()
def generate_yes_no_scores(model, tokenizer, YES_ID, NO_ID, eval_prompt):
    input_ids = tokenizer(eval_prompt, return_tensors="pt")["input_ids"].to("cuda")
    model_out = model(input_ids)
    final_output_token_logits = model_out.logits[0][-1]
    yes_score = final_output_token_logits[YES_ID].cpu().item()
    no_score = final_output_token_logits[NO_ID].cpu().item()
    return yes_score, no_score


def main(
    condition: str,
    definition: str,
    data: str,
    output: str,
    model_name: str,
    transformers_cache: str,
    device="cuda:0",
) -> None:
    """
    Main function for labelling the reports

    """
    # pretty print args
    print(f"Running Labelling for {condition} with the following args:")
    print(f"Data: {data}")
    print(f"Model: {model_name}")
    print(f"Definition: {definition}")
    print(f"Transformers Cache: {transformers_cache}")

    import os

    os.environ["TRANSFORMERS_CACHE"] = transformers_cache
    print("Using cache at: ", os.getenv("TRANSFORMERS_CACHE"))

    print(f"Loading model and tokenizer: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )
    model.config.use_cache = True
    model.half()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_bos_token=True,
        trust_remote_code=True,
        padding=True,
        padding_side="left",
        truncation=True,
        max_length=1024,
    )

    YES_TOKEN = "yes"
    NO_TOKEN = "no"
    YES_ID = tokenizer(YES_TOKEN, add_special_tokens=False).input_ids[0]
    NO_ID = tokenizer(NO_TOKEN, add_special_tokens=False).input_ids[0]

    tokenizer.pad_token = "[PAD]"

    prompt1 = f"""\
        You are a radiologist. Your job is to diagnose {condition} using a medical report. 
        Tell the truth and answer as precisely as possible.  
    """

    # TODO: load in data
    df = pd.read_csv(data, low_memory=False, index_col=0)
    print(f"Loaded in data with shape: {df.shape}")

    print("Making prompts...")
    outputs = []
    for i in tqdm(range(len(df))):
        example = df["report"].iloc[i]

        messages = [
            {"role": "system", "content": f"{prompt1}/nReport: {example}"},
            {
                "role": "user",
                "content": f"Write a summary for the above report, focusing on findings related to {condition}, according to this defintion: {definition}",
            },
        ]
        eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        gen_summary = generate_summary(model, tokenizer, eval_prompt, device)
        # Now answer the question
        messages += [
            {"role": "assistant", "content": f"{gen_summary}"},
            {
                "role": "user",
                "content": f"Based on your summary, does the patient have {condition}? Answer 'yes' for yes, 'no' for no. Only output one token after 'ANSWER: ",
            },
        ]
        eval_prompt = (
            tokenizer.apply_chat_template(messages, tokenize=False)
            + "<|assistant|>\nANSWER: "
        )
        yes_score, no_score = generate_yes_no_scores(
            model, tokenizer, YES_ID, NO_ID, eval_prompt
        )

        print("Report: ", example)
        print("Summary: ", gen_summary)
        print("Yes Score: ", yes_score)
        print("No Score: ", no_score)
        outputs.append(
            {
                "report": example,
                "summary": gen_summary,
                f"yes_score": yes_score,
                f"no_score": no_score,
            }
        )

    df_out = pd.DataFrame(outputs)
    df_out.to_csv(output, index=False)
