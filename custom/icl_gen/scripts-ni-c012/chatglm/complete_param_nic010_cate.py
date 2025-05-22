import json
import re
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import List
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llama2_prompt = """<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
<</SYS>>

{prompt} [/INST] """

alpaca_prompt = '''Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
Generate an appropriate title for the given text. The generated title must be short and include the main topic of the text. The preferred titles are under fifteen words.

{prompt}

### Response:'''

def times(n: int, length: int) -> List[List[int]]:
    if n == 1:
        return [[i] for i in range(length)]
    res_list = []
    for i in range(length):
        for lis in times(n - 1, length):
            if i not in lis:
                res_list.append([i] + lis)
    return res_list

def pack_instructions(tokenizer, template, data, definition: str = None, cate_task_style: bool = True, perm_idx=None):
    n_shots = args.n_shots
    all_perm_list = times(n_shots, 4 if cate_task_style else 20)
    inst_list, perm_list = [], []
    ignore = True
    for lis in all_perm_list:
        if perm_idx and ignore:
            if ignore and lis != perm_idx:
                continue
            else:
                ignore = False
                continue
        instruction = definition + '\n\n'
        for i in lis:
            instruction += ('Input: ' + data[i]['input'] + '\n' +
                            'Output: ' + data[i]['output'] + '\n\n')
        instruction += 'Input:'
        if template == "llama2":
            instruction = llama2_prompt.format(prompt=instruction)
        elif template == "alpaca":
            instruction = alpaca_prompt.format(prompt=instruction)
        if len(tokenizer.tokenize(instruction)) >= args.max_length:
            continue
        else:
            inst_list.append(instruction)
            perm_list.append(lis)
    return inst_list, perm_list

def icl_gen(model, tokenizer, template, cur_data, definition, cate_task_style, perm_idx=None):
    assert template in ["vanilla", "llama2", "alpaca"]
    inst_list, perm_list = pack_instructions(tokenizer, template, cur_data, definition, cate_task_style, perm_idx)

    for i, instruction in tqdm(enumerate(inst_list), total=len(inst_list)):
        responses = []
        for k in range(args.do_sample_retries):
            print("retry:", k)
            response, _ = model.chat(tokenizer, instruction, history=[], temperature=args.temperature, top_p=args.top_p, max_length = args.max_length, do_sample = args.do_sample)
            responses.append(response)）
        result_texts = list(set(responses))
        for text in result_texts:
            save_dict = {
                "inputs": instruction,
                "outputs": text,
                "perm_idx": perm_list[i]
            }
            with jsonlines.open(args.output_path, "a") as file:
                file.write(save_dict)

class CustomDataLoader(torch.utils.data.DataLoader):
    pass

def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True).half().cuda().eval()

    print("Model loaded. Beginning inference...")
    
    print("cate_task_style:",args.cate_task_style)

    with jsonlines.open(args.input_path, "r") as f:
        data = [line for line in f]

    if os.path.exists(args.output_path) and not args.resume:
        print('Error: destination output path already exists!')
        exit(-1)

    perm_idx = []
    if args.resume:
        if os.path.exists(args.output_path):
            with jsonlines.open(args.output_path, "r") as file:
                output_data = [l for l in file]
                perm_idx = output_data[-1]['perm_idx']
                print('perm_idx:', perm_idx)
                assert len(perm_idx) == args.n_shots

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    if args.cate_task_style:
        for k in range(5):
            print('task', k)
            cur_data = data[k*4:k*4+4]
            definition = cur_data[0]['full_prompt'].split('\n\n')[0]
            assert definition == cur_data[-1]['full_prompt'].split('\n\n')[0]
            icl_gen(model, tokenizer, args.template, cur_data, definition, args.cate_task_style)
    else:
        definition = data[0]['full_prompt'].split('\n\n')[0]
        assert definition == data[-1]['full_prompt'].split('\n\n')[0]
        icl_gen(model, tokenizer, args.template, data, definition, args.cate_task_style, perm_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Local path for ChatGLM model")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--do_sample", type=bool, default=True)
    parser.add_argument("--do_sample_retries", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=0.6)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--finetuning_type", type=str, default="full")
    parser.add_argument("--n_shots", type=int, default=2)
    parser.add_argument("--template", type=str, default="vanilla")
    parser.add_argument("--cate_task_style", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
