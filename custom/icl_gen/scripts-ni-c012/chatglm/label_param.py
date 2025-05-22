from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import jsonlines
from tqdm import tqdm
import ast
import argparse
from peft import PeftModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

llama2_prompt = """<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{prompt} [/INST] """

alpaca_prompt = '''Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
Generate an appropriate title for the given text. The generated title must be short and include the main topic of the text. The preferred titles are under fifteen words.

{prompt}

### Response:'''

def generate_response(model, tokenizer, instruction, args):
    with torch.no_grad():
        response, history = model.chat(tokenizer, instruction, history=[])
    return response.strip()

def main(args):
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    if args.finetuning_type == "full":
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = PeftModelForCausalLM.from_pretrained(model, args.ckpt_dir)
        model = model.merge_and_unload()
        model.to("cuda")

    model.eval()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    input_path = args.input_path
    output_path = args.output_path

    with jsonlines.open(input_path, "r") as f:
        data = [line for line in f]

    if os.path.exists(output_path):
        print('Error: destination output path already exists!')
        exit(-1)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    for i, line in tqdm(enumerate(data), total=len(data)):
        if 'inputs' in line:
            inputs = line['inputs']
        elif 'full_prompt' in line:
            inputs = line['full_prompt']
        else:
            inputs = line['definition_input']

        if args.template == "vanilla":
            instruction = inputs
        elif args.template == "llama2":
            instruction = llama2_prompt.format(prompt=inputs)
        elif args.template == "alpaca":
            instruction = alpaca_prompt.format(prompt=inputs)
        else:
            instruction = inputs

        print("========")
        print(instruction)
        print("--------")

        result_text = generate_response(model, tokenizer, instruction, args)
        print(result_text)

        save_dict = {
            "inputs": inputs,
            "targets": result_text,
        }

        with jsonlines.open(output_path, "a") as file:
            file.write(save_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="/model/llama2-3b-hf/", required=False, help=""
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="/saves/ni-c012/LLAMA2-7B/lora/qa/bs32x1x1-3ep-bf16", help=""
    )
    parser.add_argument(
        "--input_path", type=str, default="/data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-7b/ori-van/qa.train.smp001.2shot.smp3.rp1.2.json" , required=False, help=""
    )
    parser.add_argument(
        "--output_path", type=str, default="/data/ni-cus0.12/genearated-icl-naive-kmeans20-self/llama2-3b/cl_queue/qa.train.smp001.2shot.smp3.rp1.2.json", required=False, help=""
    )
    parser.add_argument("--do_sample", type=ast.literal_eval, default=False, help="")
    parser.add_argument("--top_p", type=float, default=0.6, help="")
    parser.add_argument("--temperature", type=float, default=0.9, help="")
    parser.add_argument("--max_length", type=int, default=2048, help="")
    parser.add_argument("--num_beams", type=int, default=1, help="")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="")
    parser.add_argument("--finetuning_type", type=str, default="lora")
    parser.add_argument("--template", type=str, default="llama2", help="")
    args = parser.parse_args()
    print(args)
    main(args)
