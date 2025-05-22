

import jsonlines
import os
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("  ", trust_remote_code=True)
simsce_tokenizer = AutoTokenizer.from_pretrained("model/sup-simcse-roberta-base/")

max_instruction_input_len_used=500
max_simcse_raw_input_len_used=510
min_input_len_used=20
max_input_len_used=500 
max_target_len_used=128
save_more = False # only for alpaca-7b

for cate in ["qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"]:  
    input_path = f"/data/ni-cus0.12/generated-masked-pesudo/chatglm/{cate}.train.smp001.retry3.pesudo.json"
    output_path = f"/data/ni-cus0.12/generated-masked-pesudo-filted/chatglm/ori-van/{cate}.train.smp001.1shot-mask.retry3.json"

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with jsonlines.open(input_path) as f:
        new_data = []
        for line in f:
            parts = line[0]['inputs'].split('\n\n', 1)
            if len(parts) == 2:
                definition, inputs = parts
                outputs = line[0]['outputs']
                
                if re.search('[\u4e00-\u9fa5]', inputs):
                    continue
                    
                if "____" in inputs:
                    continue
                if "=" in inputs:
                    continue
                if "___" in inputs:
                    continue
                if "_" in inputs:
                    continue
                if '\n\n' in inputs:
                    inputs = inputs.split('\n\n')[0]
                if '###' in inputs:
                    inputs = inputs.split('###')[0]
                    
                if cate == "qa":
                    keywords = ["Q:", "question:", "\nQ:", "\nquestion", "Question:", "\n\nQuestion:"]
                    for keyword in keywords:
                        inputs = inputs.replace(keyword, "\nQuestion:")
                    if "\nQuestion:" not in inputs:
                        continue

                if cate == "qg":
                    keywords = ["Q:", "question:", "\nQ:", "\nquestion", "Question:", "\nQuestion"]
                    for keyword in keywords:
                        if keyword in inputs:
                            inputs = inputs.split(keyword)[0].strip()
                
                if cate == "pe":
                    first_part = inputs.split(',', 1)[0].strip()
                    if not first_part.isdigit():
                        continue
                            
                instruction_len = len(tokenizer.encode(definition))  
                input_len = len(tokenizer.encode(inputs))
                target_len = len(tokenizer.encode(outputs))
                
                if input_len == 1:
                    continue

                if input_len > max_input_len_used:
                    continue
                if input_len < min_input_len_used:
                    continue
                if (
                    simsce_length := len(simsce_tokenizer.encode(inputs))
                ) > max_simcse_raw_input_len_used:
                    continue
                if cate == "sa":
                    if input_len > 200:
                        continue
                if cate == "qg":
                    if input_len < 100:
                        continue
                if cate == "sum":
                    if input_len < 100:
                        continue
                if cate == "trans":
                    if input_len > 80:
                        continue
                if cate == "dsg":
                    if input_len < 100:
                        continue
                if cate == "expl":
                    if input_len < 80:
                        continue

                new_data.append({'inputs': definition + '\n\n' + inputs, 'outputs': outputs})

    with jsonlines.open(output_path, 'w') as f:
        f.write_all(new_data)