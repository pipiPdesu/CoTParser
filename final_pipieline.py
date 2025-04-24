import pandas as pd
import json
import argparse
import copy
from tqdm import tqdm
import re

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch

SAMPLE_NUM = 31

def reply_by_model(system, prompt, model):
    messages =  [{
                "role": "system",
                "content": system
            },{
                "role": "user",
                "content": prompt
            },],
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    #attention_mask = input_ids['attention_mask']
    outputs = model.generate(
        input_ids,
        #attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return (tokenizer.decode(response, skip_special_tokens=True))

def batch_infer(prompt, model):
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    prompt = [[{"role":"user", "content":i}] for i in prompt]
    inputs = tokenizer.apply_chat_template(prompt, padding=True, return_tensors="pt").to(model.device)
    outputs = model.generate(
    inputs,
    max_new_tokens=512,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    do_sample=True,
    temperature=0.8,
    top_p=0.9
    )
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return generated_texts

question_parser_sentence_part = """ 
I will give you a logical question. You need to extract all the sentences from it. Return each sentence on a separate line and do not return any other content. When extracting, you must keep the original word from given text.
If you encounter two sentences that are logically connected in terms of spelling, you should merge them into one sentence for processing.
Especially, When there are parentheses listed conditions (e.g., (1), only parentheses list can be considered) within a sentence, each condition should be output as one separate sentence, with "Condition:" added at the beginning. Do not add any other prefix, such as 'Known'. 
For example, "It is known that: (1) A is a boy.C is a man,(2) B is a girl." should be output as:
Condition:(1) A is a boy.C is a man
Condition:(2) B is a girl

Do not change any other content in the extract sentence except parentheses listed conditions.
"""

question_parser_question_part = """ 
Extract useful conditions from given question in original word. When extracting, you must keep the original word from given text. If no useful condition, return None\n
"""

question_parser_justify_part = """ 
I will give you a question and a sentence.
You need to determine whether the sentence is related to the question.
If it is, return True. If the sentence is an enumerated sentence or doesn't related, return False. Do not return any other content.
Attention: 
If part of the sentence is an enumerated sentence which list some elements, such as 'A,B,C', you should return False.
"""


step_parsing = """
I will provide you with a Chain of Thought (COT) process for answering a logic problem. You need to rephrase this COT. 
Specifically, you need to identify all of the model's reasoning steps and add a "Step:" before each step.
A step is defined as a reasoning process that includes a conclusion. 
It must be coherent and logically connected, which means that they may consist of multiple sentences.
Note that:
1. The steps you restate must be derived from the original COT. When extracting, you must keep the original word from given text.
2. Only return the rephrased COT, without any additional content.
3. Steps that are related to the context are considered as one step.

Do not change any other content in the extract sentence. 
"""

reason_statement_divide = """Please extract all the conclusive statements from the given reasoning steps and add "Statement:" before each one. 
These statements refer to the conclusions drawn by the model through its own reasoning, rather than merely repeating the given conditions. 
Typically, they appear around certain adverbs or introductory words. 
When extracting, remove the reasoning part that leads to the conclusion and ensure the statement part of the original text content remains unchanged. However, if a statement contains only pronouns, please replace the pronouns with the corresponding nouns.
If no statement can be extracted, please return one word "None". 
Do not change any other content in the extract sentence. 
"""

evidence_extract = '''Now, I'm providing you with an answer and a statement extracted from that answer. Your task is to extract the reasoning process from the response that leads to this statement. 
The reasoning process is defined as the evidence that supports the statement. Typically, this evidence appears around the statement and is connected to it through adverbs, phrases, or other conjunctions. 
When extracting the evidence, do not modify any part of the original sentence, and ensure that the statement itself is not included. 
Just return the extracted evidence, prefixed with "Evidence:".
Do not change any other content in the extract sentence. Exclude the statement part from your output.'''

judge_reason =r"""
Next, I will give you a Question, a Statement, and an Evidence. You need to determine whether the Evidence supports the Statement under the context of the Question.
If the Evidence supports the Statement, please answer "True", otherwise please answer "False".
You need to give a reason for your answer, and critically analyze why it is True or False in only one simple sentence.
Return your reason and answer in follow format:
{Reason: <your reason>}
{answer: <your justification>}
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to the predictions file.", default='data/Public_Test_A.json')
    parser.add_argument("--model_path_perfix", type=str, help="Path to the model folder.", default='/root/autodl-tmp/')

    path_perfix = parser.parse_args().model_path_perfix
    model_path = path_perfix+'llama3-8b'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map="auto")
    model_statement_path = path_perfix+'statement-sft1'
    model_statement = AutoModelForCausalLM.from_pretrained(model_statement_path,torch_dtype=torch.bfloat16,device_map="auto")
    model_evidence_path = path_perfix+'evidence-sft1'
    model_evidence = AutoModelForCausalLM.from_pretrained(model_evidence_path ,torch_dtype=torch.bfloat16,device_map="auto")
    model_reason_path = path_perfix+'reason-sft1'
    model_reason = AutoModelForCausalLM.from_pretrained(model_reason_path,torch_dtype=torch.bfloat16,device_map="auto")

    data = json.load(open(parser.parse_args().input_file))



    ans = copy.deepcopy(data)
    for i in range(len(data)):
        ## question part
        
        context = data[i]['question'].split('\n')
        if len(context) != 6:
            context = [data[i]['question']] + [data[i]['question']]
        
        context[0] = context[0].replace("?", ":")
        conditions = []
        #question extraction part
        reply = reply_by_model(question_parser_sentence_part, context[0], model=model)
        sentence = reply.split('\n')
        extract_sentence = set()
        for sec in sentence:
            sec = sec.replace("?", ":").strip(" ")
            prompt = f"{sec}"
            if sec == '' or "Here are the extracted sentences" in sec or sec[-1] == ":":
                continue
            if "Condition:" in sec:
                sec = sec.replace("Condition:", "").strip(" ")
                if sec[0] == "(":
                    sec = sec[3:]
                extract_sentence.add(sec.strip(" "))
            else:
                    if sec[0] == "(":
                        sec = sec[3:]
                    extract_sentence.add(sec)
        if 'If' in context[1] or 'Since' in context[1]:
                pattern = r"^(.*?),[^,]*$"
                match = re.match(pattern, context[1])
                if match:
                    result = match.group(1)
                    conditions.append(result.replace("Condition:", "").replace("Since", "").replace("If", "").strip(" "))
        conditions = conditions + list(extract_sentence)
        ans[i]['question_parsing'] = conditions
        

        ## cot part
        #driven step
        reply = reply_by_model(step_parsing, data[i]['cot'].replace('\n', ''), model=model)
        sentence = reply.split('\n')
        ans[i]['cot_parsing'] = []
        for sec in sentence:
            if sec.startswith('Step:'):
                #statement part
                divide = reply_by_model("",reason_statement_divide+'\n'+sec, model=model_statement).split('\n')
                for state in divide:
                    if "Statement" in state:
                        statement = state.replace("Statement:", "").strip(" ")
                        if statement == '' or statement == 'None':
                            continue
                        if statement.strip(".").lower() in data[i]['question'].lower():
                            continue
                        if len(statement) <= 1:
                            continue
                        if statement[-1] not in ['.', '!', '?', ',', ';']:
                            statement = statement + '.'
                        ans[i]['cot_parsing'].append({})
                        ans[i]['cot_parsing'][-1]['statement'] = statement
                        #evidence
                        temp_evid = reply_by_model("", evidence_extract+'\nAnswer: '+ans[i]['cot']+'\nStatement:'+statement, model=model_evidence)
                        evid = temp_evid.replace("Evidence:", "").strip(" ")
                        ans[i]['cot_parsing'][-1]['evidence'] = evid

                        ## reason
                        prompt = judge_reason+'Question: ' + data[i]['question'] + '\nStatement: ' + statement+ '\nEvidence: ' + evid
                        reply = batch_infer([prompt]*SAMPLE_NUM, model=model_reason)
                        reply = [seq.split('answer')[-1].strip(":").strip("}").strip(" ").lower() for seq in reply]
                        justify = 'true' if reply.count('true')>SAMPLE_NUM//2 else 'false'
                        ans[i]['cot_parsing'][-1]['Verification'] = justify
                
                        
        #print(ans[i]['cot_parsing'])
    
    with open("results.json", "w", encoding="utf-8") as file:
    # 将整个数据结构写入文件，格式化为 JSON 格式
        json.dump(ans, file, ensure_ascii=False, indent=4)

    

    
