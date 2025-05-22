import json
import re
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer, AutoModel


class InstructionAwareGenerator:
    def __init__(self, args):
        self.args = args
        self._init_model(args)
        self.instruction_cache = {}
        self.task = Path(args.input_path).name.split('.')[0]

    def _init_model(self, args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(args.generate_model_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        self.tokenizer.padding_side = "left"

        self.model = AutoModel.from_pretrained(
            args.generate_model_path,
            trust_remote_code=True
        ).half().cuda().eval()  

    def get_mask_prompt_template(self, task: str, masked_text, masked_example, example) -> str:
        default_template = (
            """Replace `____` in the context while keeping them logically consistent. Make sure the format is consistent with the example. Replace only with words or numbers.

            Here is an example:

            Input: 
            {masked_example}

            Output: 
            {example}

            Now, process the following input:

            Input: 
            {masked_text}

            Output: 
            """)
        return default_template.format(masked_example=masked_example, example=example, masked_text=masked_text)  

    def extract_instruction(self, full_prompt: str) -> str:
        separators = ["\n\n"]
        for sep in separators:
            if sep in full_prompt:
                parts = full_prompt.split(sep, 1)
                self.instruction_cache = parts[0] + sep  
                return self.instruction_cache
        return full_prompt.split("{input}")[0] if "{input}" in full_prompt else full_prompt

    def contextual_mask(self, context: str, mask_ratio=(0.1, 0.5)) -> str:
        protected_terms = {"Context:", "Question:", "\"", "[", "]", ","}
        tokens = context.strip().split()
        n = len(tokens)
        if n < 3:
            return context
            
        if n < 20:
            max_ratio = 0.2  
        elif n < 100:
            max_ratio = 0.4  
        else:
            max_ratio = 0.6  
                   
        actual_ratio = random.uniform(mask_ratio[0], min(mask_ratio[1], max_ratio))
        
        all_indices = set(range(n))
        protected_indices = {i for i, w in enumerate(tokens) if w in protected_terms}
        candidate_indices = list(all_indices - protected_indices)
        
        keep_num = int(len(candidate_indices) * (1 - actual_ratio))
        keep_non_protected = set(random.sample(candidate_indices, k=keep_num))
        keep_indices = protected_indices | keep_non_protected
        
        masked = []
        prev_masked = False
        for i, word in enumerate(tokens):
            if word in protected_terms:
                masked.append(word)
                prev_masked = False
            elif i in keep_indices or self._should_retain(word):
                masked.append(word)
                prev_masked = False
            else:
                if not prev_masked:
                    masked.append("____")
                prev_masked = True
        
        text = ' '.join(masked)
        return re.sub(r'____( ____)+', '____', text)

    def generate_completion(self, masked_text, masked_example, example, num_samples=1):
        template = self.get_mask_prompt_template(self.task, masked_text=masked_text, masked_example=masked_example, example=example)
        print("template: ", template)

        results = []
        for _ in range(num_samples):
            response, _ = self.model.chat(self.tokenizer, template, history=[])
            results.append(response)
        
        return results  

    def mask_and_generate(self, samples: List[Dict], num_samples=1) -> List[List[Dict]]:
        results = []
        for i in range(self.args.retry):
            for idx, sample in enumerate(samples):
                samples_left = samples[:idx] + samples[idx+1:]
                for example in samples_left:
                    instruction = self.extract_instruction(sample["full_prompt"])
                    masked_input = self.contextual_mask(sample["input"])
                    example_input = example["input"]
                    masked_expample_input = self.contextual_mask(example["input"])
                    
                    print("check masked_input", masked_input)
                    
                    gen_inputs = self.generate_completion(masked_input, masked_expample_input, example_input, num_samples)
                
                    sample_results = []
                    for gen_input in gen_inputs:
                        full_prompt = f"{instruction}{gen_input}"
                        sample_results.append({"inputs": full_prompt, "outputs": " "})
                    results.append(sample_results)
        return results

    def _should_retain(self, token: str) -> bool:
        return any([
            re.match(r'^-?\d+\.?\d*$', token),
            token.istitle() and len(token) > 3,
            re.match(r'^[A-Z]{3,}$', token),
            token in {'not', 'no', 'never'}
        ])

def main(args):
    with open(args.input_path, "r", encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    
    generator = InstructionAwareGenerator(args)
    pseudo_samples = generator.mask_and_generate(samples, args.num_return_sentence)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample_result in pseudo_samples:
            f.write(json.dumps(sample_result, ensure_ascii=False) + "\n")
    print("Finish!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_model_path", type=str, default="your_model_path")
    parser.add_argument("--input_path", type=str, default="your_input.json")
    parser.add_argument("--output_path", type=str, default="your_output.json")
    parser.add_argument("--num_return_sentence", type=int, default=1)
    parser.add_argument("--retry", type=int, default=20)
    args = parser.parse_args()
    
    main(args)
