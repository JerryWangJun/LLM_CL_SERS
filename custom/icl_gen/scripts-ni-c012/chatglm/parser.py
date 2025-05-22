

import jsonlines
import os
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/model/chatglm/",trust_remote_code=True)
simsce_tokenizer = AutoTokenizer.from_pretrained("/model/sup-simcse-roberta-base/")

max_instruction_input_len_used=500
max_simcse_raw_input_len_used=510
max_input_len_used=800 
max_target_len_used=128
save_more = False # only for alpaca-7b

for cate in ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]: #, 

    input_path = f'/data/ni-cus0.12/genearated-icl-naive/chatglm/ori/{cate}.train.smp001.2shot.smp3.rp1.2.json'
    output_path = f'/data/ni-cus0.12/genearated-icl-naive-parsed-filtered/chatglm/ori-van/{cate}.train.smp001.2shot.smp3.rp1.2.json'

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with jsonlines.open(input_path) as f:
        new_data = []
        reject_stats = {} 
        
        for line_idx, line in enumerate(f):
            definition = line['inputs'].split('\n\nInput:')[0]
            instruction_len = len(tokenizer.encode(definition))
            insts = line['outputs'].split('Input:')
            
            for inst in insts:
                pair = inst.split('Output:')
                reject_reason = None  
                

                if len(pair) != 2:
                    if not (save_more and len(pair) == 1):
                        reject_reason = "invalid_output_format"

                if len(pair) == 2:
                    inputs, outputs = pair[0].strip(), pair[1].strip()
                    
                    input_len = len(tokenizer.encode(inputs))
                    target_len = len(tokenizer.encode(outputs))
    

                    if input_len == 1:
                        reject_reason = "empty_input"
                    elif target_len == 1 and not save_more:
                        reject_reason = "empty_output"
                    elif (
                        max_instruction_input_len_used
                        and input_len + instruction_len > max_instruction_input_len_used
                    ):
                        reject_reason = "instruction_input_too_long"
                    elif input_len > max_input_len_used or target_len > max_target_len_used:
                        reject_reason = "input_or_target_exceed_length"
                    elif (
                        len(simsce_tokenizer.encode(inputs)) > max_simcse_raw_input_len_used
                    ):
                        reject_reason = "simcse_input_too_long"
    

                if reject_reason:
                    reject_stats[reject_reason] = reject_stats.get(reject_reason, 0) + 1
                    print(f"Rejected sample (line {line_idx}): {reject_reason}")
                    continue
                    
                new_data.append({'inputs': definition + '\n\n' + inputs, 'outputs': outputs})
    
    print("\n=== Filter Statistics ===")
    for reason, count in reject_stats.items():
        print(f"{reason}: {count} samples filtered")



    with jsonlines.open(output_path, 'w') as f:
        f.write_all(new_data)