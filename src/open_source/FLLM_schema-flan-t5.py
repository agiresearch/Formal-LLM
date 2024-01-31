from torch.utils.data import DataLoader
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import os
os.chdir('../')
import sys
sys.path.append('../')
print(sys.path)
from generate_model_seq import SeqGen
import torch.optim as optim
from general_dataset import GeneralDataset
from agi_utils import *
from combine_model_seq import SeqCombine

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--num_seq", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--decay_rate", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--accumulate_steps", type=int, default=1)
parser.add_argument("--warm_up_proportion", type=float, default=0.1)
parser.add_argument('--bl_dec', type=float, default=0.8)
parser.add_argument('--pos_reward_coef', type=float, default=1.0)
parser.add_argument("--cache_dir", type=str, default='./')
parser.add_argument("--log_name", type=str, default='./0.txt')
parser.add_argument("--ckp_dir", type=str, default='./')

args = parser.parse_args("")

"""
load training and test datasets
"""
data_path = "./openagi_data/"

task_discriptions = txt_loader(data_path + "/task_description.txt")
training_task_idx = [7,20,30,40,50,60, 0, 105, 110, 116, 117, 175, 182]
test_task_idx = [2, 3, 10, 15, 20, 35, 45, 55, 65, 70, 90, 106, 112, 115, 118, 177, 179]
training_dataloaders = []
test_dataloaders = []

for i in training_task_idx:
    dataset = GeneralDataset(i, data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    training_dataloaders.append(dataloader)
    
for j in test_task_idx:
    dataset = GeneralDataset(j,data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    test_dataloaders.append(dataloader)
    
training_tasks = [task_discriptions[i].strip() for i in training_task_idx]
test_tasks = [task_discriptions[j].strip() for j in test_task_idx]
print(training_tasks)
print(test_tasks)

import numpy as np
from IPython.utils import io
import random
from tqdm import tqdm
from evaluate import load
from torchvision import transforms
from transformers import AutoModel, AutoFeatureExtractor
from torchmetrics.multimodal import CLIPScore

clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

# Load a pre-trained Vision Transformer model and its feature extractor
vit_ckpt = "nateraw/vit-base-beans"
vit = AutoModel.from_pretrained(vit_ckpt, cache_dir=args.cache_dir)
vit.eval()
vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt, cache_dir=args.cache_dir)

f = transforms.ToPILImage()
bertscore = load("bertscore")

args.device_list = ["cuda:3", "cuda:2", "cuda:1", "cpu"]
args.huggingface_cache = args.cache_dir
seqCombination = SeqCombine(args)

from undecorated import undecorated
from finetune.utils import construct_optimizer
from types import MethodType
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch
from peft import PeftModel, PeftModelForCausalLM, prepare_model_for_int8_training, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig, AutoFeatureExtractor

device = 'cuda:0'

tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = T5ForConditionalGeneration.from_pretrained(args.ckp_dir, cache_dir=args.cache_dir, device_map='auto')

model.gradient_checkpointing_enable()
model = prepare_model_for_int8_training(model)

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
lora_target_modules = ["q_proj","v_proj"]

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

generate_with_grad = undecorated(model.generate)
model.generate_with_grad = MethodType(generate_with_grad, model)

model = get_peft_model(model, config)
model.print_trainable_parameters()
model.config.use_cache = False

seqGen = SeqGen(model, tokenizer, device)
optimizer, scheduler = construct_optimizer(args, seqGen.model, 20)

import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args.log_name, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(vars(args))

task2type = dict()
for index in range(0, 6):
    for task in seqGen.candidates[index]['task_list']:
        task2type[task] = index
baseline = []

for i, task_description in enumerate(training_tasks):
    with torch.no_grad():
        logging.info(f'Solving tasks: {i + 1} / {len(training_tasks)}')
        auto_stack = []
        action = []
        model_stack = []
        necessary_i2t = set([x[:-1] for x in seqGen.candidates[1]['task_list']])
        necessary_t2i = set([x[:-1] for x in seqGen.candidates[3]['task_list']])
                
        if len(training_dataloaders[i].dataset.output_file_type) > 0 and training_dataloaders[i].dataset.output_file_type[0] == 'text':
            auto_stack.append('T')
            model_stack.append('final text')
        elif training_task_idx[i] in [105, 106] or training_dataloaders[i].dataset.output_file_type[0] == 'image':
            auto_stack.append('I')
            model_stack.append('final image')
        else:
            raise NotImplementedError

        while len(auto_stack) > 0:
            stack_top = auto_stack.pop()
            tool_cnt = 0
            prompt = f'You will help me generate a plan for the Problem: "{task_description}" by answering a series of my questions.\n'

            prompt += '\nCurrect Progress: \n'
            for j, step in enumerate(action):
                if j > 0:
                    prompt += f'Step (n-{j}): Use {step};\n'
                else:
                    prompt += f'Step n: Use {step};\n'
            if len(action) == 0:
                prompt += 'Step n: ?\n'
            else:
                prompt += f'Step (n-{len(action)}): ?\n'
            
            prompt += f'\nTo get the {model_stack.pop()}, we have the following choices:\n'
            model_choice = [None]
            
            if stack_top == 'T':
                available = [1, 2, 4, 5] # i2t, t2t, tt2t, it2t
                if 'image' not in training_dataloaders[i].dataset.input_file_type and len(necessary_t2i) <= auto_stack.count('I'):
                    available = [2, 4] # remove i2t, it2t
                if 'text' not in training_dataloaders[i].dataset.input_file_type and len(necessary_i2t) < 2 + auto_stack.count('T'):
                    available.remove(4)  # remove tt2t
                for index in available: 
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0  and 'text' in training_dataloaders[i].dataset.input_file_type:
                    if training_dataloaders[i].dataset.input_file_type.count('text') == 1:
                        tool_cnt += 1
                        prompt += f'{tool_cnt}: Input Text.\n'
                        model_choice.append('Input Text.')
                    else:
                        for t in range(training_dataloaders[i].dataset.input_file_type.count('text')):
                            tool_cnt += 1
                            model_choice.append(f'Input Text {t}.')
                            prompt += f'{tool_cnt}: Input Text {t};\n'
            
            elif stack_top == 'I':
                available = [0, 3] # i2i, t2i
                if 'text' not in training_dataloaders[i].dataset.input_file_type and len(necessary_i2t) <= auto_stack.count('T'):
                    available = [0]  # remove t2i
                for index in available:
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0  and 'image' in training_dataloaders[i].dataset.input_file_type:
                    if training_dataloaders[i].dataset.input_file_type.count('image') == 1:
                        tool_cnt += 1
                        prompt += f'{tool_cnt}: Input Image.\n'
                        model_choice.append('Input Image.')
                    else:
                        for t in range(training_dataloaders[i].dataset.input_file_type.count('image')):
                            tool_cnt += 1
                            prompt += f'{tool_cnt}: Input Image {t};\n'
                            model_choice.append(f'Input Image {t}.')
                
            else:
                raise NotImplementedError
            
            prompt += f'\nYour answer should be only an integer, referring to desired choice.\n' 
            logging.info(prompt)
    
            input_ids = seqGen.tokenizer.batch_encode_plus([prompt], padding="longest", return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(device)
            output = seqGen.model.generate(input_ids, max_length=30, min_length=1, return_dict_in_generate=True, output_scores=True, 
                                             output_hidden_states=True, renormalize_logits=True, temperature=1, top_k=5, top_p=0.5,
                                             num_return_sequences=1)
            output_ids = output["sequences"][:, 1:]
            output_sequence = [s.replace("<pad>", "").replace("</s>", "") for s in seqGen.tokenizer.batch_decode(output_ids)]
            logging.info(output_sequence)
            
            scores = output['scores']
            log_probs = []
            for t in range(1, tool_cnt + 1):
                out_id = torch.tensor([seqGen.tokenizer.encode(str(t))])
                logprob = 0
                length = out_id.size(-1)
                for l in range(length):
                    score = scores[l][0]
                    logprob += score[out_id[0][l]]
                    if seqGen.tokenizer.decode(out_id[0][l]) == "</s>":
                        continue
                log_probs.append(logprob)
            logging.info(log_probs)
    
            choice = torch.argmax(torch.stack(log_probs).detach())
            choice = choice.item() + 1
            logging.info(choice)
            
            if choice <= 0 or choice > tool_cnt:
                logging.info('Answer Out of Range')
                auto_stack.append(choice)
                break
    
            choice = model_choice[choice]
            action.append(choice[:-1])
            if choice in task2type:
                index = task2type[choice]
                if index == 0:  # i2i
                    auto_stack.append('I')
                    model_stack.append(f'input image of "{action[-1]}"')
                elif index == 1:  # i2t
                    auto_stack.append('I')
                    model_stack.append(f'input image of "{action[-1]}"')
                    assert action[-1] in necessary_i2t
                    necessary_i2t.remove(action[-1])
                elif index == 2:  # t2t
                    auto_stack.append('T')
                    model_stack.append(f'input text of "{action[-1]}"')
                elif index == 3:  # t2i
                    auto_stack.append('T')
                    model_stack.append(f'input text of "{action[-1]}"')
                    assert action[-1] in necessary_t2i
                    necessary_t2i.remove(action[-1])
                elif index == 4:  # tt2t
                    auto_stack += ['T', 'T']
                    model_stack.append(f'second input text of "{action[-1]}"')
                    model_stack.append(f'first input text of "{action[-1]}"')
                elif index == 5: # it2t
                    auto_stack += ['I', 'T']
                    model_stack.append(f'input image of "{action[-1]}"')
                    model_stack.append(f'input text of "{action[-1]}"')
                else:
                    raise NotImplementedError
        
        module_list = action[::-1]
        logging.info(f'Module Sequence: {module_list}')
        
        if len(auto_stack) > 0:
            logging.info('Invalid or Too long plan')
            ave_task_reward = 0
        else:
            seqCombination.construct_module_tree(module_list)
            task_rewards = []
            for idx, batch in enumerate(tqdm(training_dataloaders[i])):
                inputs = [list(input_data) for input_data in batch['input']]
                predictions = seqCombination.run_module_tree(module_list, inputs, training_dataloaders[i].dataset.input_file_type)
    
                if 0 <= training_task_idx[i] <= 14:
                    outputs = list(batch['output'][0])
                    dist = image_similarity(predictions, outputs, vit, vit_extractor)
                    task_rewards.append(dist/100)
                elif 15 <= training_task_idx[i] <= 104 or 107 <= training_task_idx[i] <= 184:
                    outputs = list(batch['output'][0])
                    f1 = np.mean(txt_eval(predictions, outputs, bertscore, device="cuda:0"))
                    task_rewards.append(f1)
                else:
                    predictions = [pred for pred in predictions]
                    inputs = [text for text in inputs[0]]
                    score = clip_score(predictions, inputs)
                    task_rewards.append(score.detach()/100)
            ave_task_reward = np.mean(task_rewards)
            logging.info("Average reward on current task: " + str(ave_task_reward))
            seqCombination.close_module_seq()
            
        baseline.append(ave_task_reward)

print(np.mean(baseline))

epochs = args.epochs
epsilon = args.epsilon
decay_rate = args.decay_rate
random.seed(args.seed)

task2type = dict()
for index in range(0, 6):
    for task in seqGen.candidates[index]['task_list']:
        task2type[task] = index

for e in range(epochs):
    epoch_total_loss = 0.0
    rewards = []
    logging.info(f'num of epoch {e + 1}')
    optimizer.zero_grad()
    for i, task_description in enumerate(training_tasks):
        total_loss = 0
        total_log_prob = 0
        logging.info(f'Solving tasks: {i + 1} / {len(training_tasks)}')
        auto_stack = []
        action = []
        model_stack = []
        necessary_i2t = set([x[:-1] for x in seqGen.candidates[1]['task_list']])
        necessary_t2i = set([x[:-1] for x in seqGen.candidates[3]['task_list']])
                
        if len(training_dataloaders[i].dataset.output_file_type) > 0 and training_dataloaders[i].dataset.output_file_type[0] == 'text':
            auto_stack.append('T')
            model_stack.append('final text')
        elif training_task_idx[i] in [105, 106] or training_dataloaders[i].dataset.output_file_type[0] == 'image':
            auto_stack.append('I')
            model_stack.append('final image')
        else:
            raise NotImplementedError

        while len(auto_stack) > 0:
            stack_top = auto_stack.pop()
            tool_cnt = 0
            prompt = f'You will help me generate a plan for the Problem: "{task_description}" by answering a series of my questions.\n'
            
            prompt += '\nCurrect Progress: \n'
            for j, step in enumerate(action):
                if j > 0:
                    prompt += f'Step (n-{j}): Use {step};\n'
                else:
                    prompt += f'Step n: Use {step};\n'
            if len(action) == 0:
                prompt += 'Step n: ?\n'
            else:
                prompt += f'Step (n-{len(action)}): ?\n'
            
            prompt += f'\nTo get the {model_stack.pop()}, we have the following choices:\n'
            model_choice = [None]
            
            if stack_top == 'T':
                available = [1, 2, 4, 5] # i2t, t2t, tt2t, it2t
                if 'image' not in training_dataloaders[i].dataset.input_file_type and len(necessary_t2i) <= auto_stack.count('I'):
                    available = [2, 4] # remove i2t, it2t
                if 'text' not in training_dataloaders[i].dataset.input_file_type and len(necessary_i2t) < 2 + auto_stack.count('T'):
                    available.remove(4)  # remove tt2t
                for index in available: 
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0  and 'text' in training_dataloaders[i].dataset.input_file_type:
                    if training_dataloaders[i].dataset.input_file_type.count('text') == 1:
                        tool_cnt += 1
                        prompt += f'{tool_cnt}: Input Text.\n'
                        model_choice.append('Input Text.')
                    else:
                        for t in range(training_dataloaders[i].dataset.input_file_type.count('text')):
                            tool_cnt += 1
                            model_choice.append(f'Input Text {t}.')
                            prompt += f'{tool_cnt}: Input Text {t};\n'
            
            elif stack_top == 'I':
                available = [0, 3] # i2i, t2i
                if 'text' not in training_dataloaders[i].dataset.input_file_type and len(necessary_i2t) <= auto_stack.count('T'):
                    available = [0]  # remove t2i
                for index in available:
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0  and 'image' in training_dataloaders[i].dataset.input_file_type:
                    if training_dataloaders[i].dataset.input_file_type.count('image') == 1:
                        tool_cnt += 1
                        prompt += f'{tool_cnt}: Input Image.\n'
                        model_choice.append('Input Image.')
                    else:
                        for t in range(training_dataloaders[i].dataset.input_file_type.count('image')):
                            tool_cnt += 1
                            prompt += f'{tool_cnt}: Input Image {t};\n'
                            model_choice.append(f'Input Image {t}.')
                
            else:
                raise NotImplementedError

            prompt += f'\nYour answer should be only an integer, referring to desired choice.\n' 
            logging.info(prompt)
    
            input_ids = seqGen.tokenizer.batch_encode_plus([prompt], padding="longest", return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(device)
            output = seqGen.model.generate_with_grad(input_ids, max_length=30, min_length=1, return_dict_in_generate=True, output_scores=True, 
                                                     output_hidden_states=True, renormalize_logits=True, temperature=1, top_k=5, top_p=0.5,
                                                     num_return_sequences=1)
            output_ids = output["sequences"][:, 1:]
            output_sequence = [s.replace("<pad>", "").replace("</s>", "") for s in seqGen.tokenizer.batch_decode(output_ids)]
            logging.info(output_sequence)
            
            scores = output['scores']
            log_probs = []
            for t in range(1, tool_cnt + 1):
                out_id = torch.tensor([seqGen.tokenizer.encode(str(t))])
                logprob = 0
                length = out_id.size(-1)
                for l in range(length):
                    score = scores[l][0]
                    logprob += score[out_id[0][l]]
                    if seqGen.tokenizer.decode(out_id[0][l]) == "</s>":
                        continue
                log_probs.append(logprob)
            logging.info(log_probs)
    
            random_num = random.random()
            logits = torch.stack(log_probs)
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            if random_num >= epsilon:
                choice = torch.argmax(logits)
            else:
                choice = torch.distributions.Categorical(logits=logits).sample() 
            total_log_prob -= logits[choice]
            choice = choice.item() + 1
            logging.info(choice)

            if choice <= 0 or choice > tool_cnt:
                logging.info('Answer Out of Range')
                break
    
            choice = model_choice[choice]
            action.append(choice[:-1])
            if choice in task2type:
                index = task2type[choice]
                if index == 0:  # i2i
                    auto_stack.append('I')
                    model_stack.append(f'input image of "{action[-1]}"')
                elif index == 1:  # i2t
                    auto_stack.append('I')
                    model_stack.append(f'input image of "{action[-1]}"')
                    assert action[-1] in necessary_i2t
                    necessary_i2t.remove(action[-1])
                elif index == 2:  # t2t
                    auto_stack.append('T')
                    model_stack.append(f'input text of "{action[-1]}"')
                elif index == 3:  # t2i
                    auto_stack.append('T')
                    model_stack.append(f'input text of "{action[-1]}"')
                    assert action[-1] in necessary_t2i
                    necessary_t2i.remove(action[-1])
                elif index == 4:  # tt2t
                    auto_stack += ['T', 'T']
                    model_stack.append(f'second input text of "{action[-1]}"')
                    model_stack.append(f'first input text of "{action[-1]}"')
                elif index == 5: # it2t
                    auto_stack += ['I', 'T']
                    model_stack.append(f'input image of "{action[-1]}"')
                    model_stack.append(f'input text of "{action[-1]}"')
                else:
                    raise NotImplementedError
        
        module_list = action[::-1]
        logging.info(f'Module Sequence: {module_list}')
        
        if len(auto_stack) > 0:
            logging.info('Invalid or Too long plan')
        else:
            with torch.no_grad():
                seqCombination.construct_module_tree(module_list)
                task_rewards = []
                for idx, batch in enumerate(tqdm(training_dataloaders[i])):
                    inputs = [list(input_data) for input_data in batch['input']]
                    predictions = seqCombination.run_module_tree(module_list, inputs, training_dataloaders[i].dataset.input_file_type)
        
                    if len(training_dataloaders[i].dataset.output_file_type) > 0 and 0 <= training_task_idx[i] <= 14:
                        outputs = list(batch['output'][0])
                        dist = image_similarity(predictions, outputs, vit, vit_extractor)
                        task_rewards.append(dist/100)
                    elif 15 <= training_task_idx[i] <= 104 or 107 <= training_task_idx[i] <= 184:
                        outputs = list(batch['output'][0])
                        f1 = np.mean(txt_eval(predictions, outputs, bertscore, device="cuda:0"))
                        task_rewards.append(f1)
                    else:
                        predictions = [pred for pred in predictions]
                        inputs = [text for text in inputs[0]]
                        score = clip_score(predictions, inputs)
                        task_rewards.append(score.detach()/100)
                ave_task_reward = np.mean(task_rewards)
                logging.info("Average reward on current task: " + str(ave_task_reward))
                rewards.append(ave_task_reward)
                seqCombination.close_module_seq()
        total_loss += total_log_prob * (ave_task_reward - baseline[i])
        if total_loss > 0.0:
            total_loss *= args.pos_reward_coef
        total_loss.backward()
        baseline[i] -= (1 - args.bl_dec) * (baseline[i] - ave_task_reward)
        epoch_total_loss += total_loss.detach()
        
    avg_reward = np.mean(rewards)
    logging.info(f"Average reward: {avg_reward}")
    logging.info(f"Loss: {epoch_total_loss.item()}")
    
    optimizer.step()
    scheduler.step()
    epsilon *= decay_rate

seqGen.model.save_pretrained(args.ckp_dir)

rewards = []
clips = []
berts = []
similairies = []

task2type = dict()
for index in range(0, 6):
    for task in seqGen.candidates[index]['task_list']:
        task2type[task] = index

for i, task_description in enumerate(test_tasks):
    with torch.no_grad():
        logging.info(f'Solving tasks: {i + 1} / {len(test_tasks)}')
        auto_stack = []
        action = []
        model_stack = []
        necessary_i2t = set([x[:-1] for x in seqGen.candidates[1]['task_list']])
        necessary_t2i = set([x[:-1] for x in seqGen.candidates[3]['task_list']])
                
        if len(test_dataloaders[i].dataset.output_file_type) > 0 and test_dataloaders[i].dataset.output_file_type[0] == 'text':
            auto_stack.append('T')
            model_stack.append('final text')
        elif test_task_idx[i] in [105, 106] or test_dataloaders[i].dataset.output_file_type[0] == 'image':
            auto_stack.append('I')
            model_stack.append('final image')
        else:
            raise NotImplementedError

        while len(auto_stack) > 0:
            stack_top = auto_stack.pop()
            tool_cnt = 0
            prompt = f'You will help me generate a plan for the Problem: "{task_description}" by answering a series of my questions.\n'
            
            prompt += '\nCurrect Progress: \n'
            for j, step in enumerate(action):
                if j > 0:
                    prompt += f'Step (n-{j}): Use {step};\n'
                else:
                    prompt += f'Step n: Use {step};\n'
            if len(action) == 0:
                prompt += 'Step n: ?\n'
            else:
                prompt += f'Step (n-{len(action)}): ?\n'
            
            prompt += f'\nTo get the {model_stack.pop()}, we have the following choices:\n'
            model_choice = [None]
            
            if stack_top == 'T':
                available = [1, 2, 4, 5] # i2t, t2t, tt2t, it2t
                if 'image' not in test_dataloaders[i].dataset.input_file_type and len(necessary_t2i) <= auto_stack.count('I'):
                    available = [2, 4] # remove i2t, it2t
                if 'text' not in test_dataloaders[i].dataset.input_file_type and len(necessary_i2t) < 2 + auto_stack.count('T'):
                    available.remove(4)  # remove tt2t
                for index in available: 
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0  and 'text' in test_dataloaders[i].dataset.input_file_type:
                    if test_dataloaders[i].dataset.input_file_type.count('text') == 1:
                        tool_cnt += 1
                        prompt += f'{tool_cnt}: Input Text.\n'
                        model_choice.append('Input Text.')
                    else:
                        for t in range(test_dataloaders[i].dataset.input_file_type.count('text')):
                            tool_cnt += 1
                            model_choice.append(f'Input Text {t}.')
                            prompt += f'{tool_cnt}: Input Text {t};\n'
            
            elif stack_top == 'I':
                available = [0, 3] # i2i, t2i
                if 'text' not in test_dataloaders[i].dataset.input_file_type and len(necessary_i2t) <= auto_stack.count('T'):
                    available = [0]  # remove t2i
                for index in available:
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0  and 'image' in test_dataloaders[i].dataset.input_file_type:
                    if test_dataloaders[i].dataset.input_file_type.count('image') == 1:
                        tool_cnt += 1
                        prompt += f'{tool_cnt}: Input Image.\n'
                        model_choice.append('Input Image.')
                    else:
                        for t in range(test_dataloaders[i].dataset.input_file_type.count('image')):
                            tool_cnt += 1
                            prompt += f'{tool_cnt}: Input Image {t};\n'
                            model_choice.append(f'Input Image {t}.')
                
            else:
                raise NotImplementedError

            prompt += f'\nYour answer should be only an integer, referring to desired choice.\n' 
            logging.info(prompt)
    
            input_ids = seqGen.tokenizer.batch_encode_plus([prompt], padding="longest", return_tensors="pt")["input_ids"]
            input_ids = input_ids.to(device)
            output = seqGen.model.generate(input_ids, max_length=30, min_length=1, return_dict_in_generate=True, output_scores=True, 
                                             output_hidden_states=True, renormalize_logits=True, temperature=1, top_k=5, top_p=0.5,
                                             num_return_sequences=1)
            output_ids = output["sequences"][:, 1:]
            output_sequence = [s.replace("<pad>", "").replace("</s>", "") for s in seqGen.tokenizer.batch_decode(output_ids)]
            logging.info(output_sequence)
            scores = output['scores']
            log_probs = []
            for t in range(1, tool_cnt + 1):
                out_id = torch.tensor([seqGen.tokenizer.encode(str(t))])
                logprob = 0
                length = out_id.size(-1)
                for l in range(length):
                    score = scores[l][0]
                    logprob += score[out_id[0][l]]
                    if seqGen.tokenizer.decode(out_id[0][l]) == "</s>":
                        continue
                log_probs.append(logprob)
            logging.info(log_probs)
    
            choice = torch.argmax(torch.stack(log_probs).detach())
            choice = choice.item() + 1
            logging.info(choice)
            
            if choice <= 0 or choice > tool_cnt:
                logging.info('Answer Out of Range')
                auto_stack.append(choice)
                break
    
            choice = model_choice[choice]
            action.append(choice[:-1])
            if choice in task2type:
                index = task2type[choice]
                if index == 0:  # i2i
                    auto_stack.append('I')
                    model_stack.append(f'input image of "{action[-1]}"')
                elif index == 1:  # i2t
                    auto_stack.append('I')
                    model_stack.append(f'input image of "{action[-1]}"')
                    assert action[-1] in necessary_i2t
                    necessary_i2t.remove(action[-1])
                elif index == 2:  # t2t
                    auto_stack.append('T')
                    model_stack.append(f'input text of "{action[-1]}"')
                elif index == 3:  # t2i
                    auto_stack.append('T')
                    model_stack.append(f'input text of "{action[-1]}"')
                    assert action[-1] in necessary_t2i
                    necessary_t2i.remove(action[-1])
                elif index == 4:  # tt2t
                    auto_stack += ['T', 'T']
                    model_stack.append(f'second input text of "{action[-1]}"')
                    model_stack.append(f'first input text of "{action[-1]}"')
                elif index == 5: # it2t
                    auto_stack += ['I', 'T']
                    model_stack.append(f'input image of "{action[-1]}"')
                    model_stack.append(f'input text of "{action[-1]}"')
                else:
                    raise NotImplementedError
        
        module_list = action[::-1]
        logging.info(f'Module Sequence: {module_list}')
        
        if len(auto_stack) > 0:
            logging.info('Invalid or Too long plan')
            ave_task_reward = 0
        else:
            seqCombination.construct_module_tree(module_list)
            task_rewards = []
            for idx, batch in enumerate(tqdm(test_dataloaders[i])):
                inputs = [list(input_data) for input_data in batch['input']]
                predictions = seqCombination.run_module_tree(module_list, inputs, test_dataloaders[i].dataset.input_file_type)
    
                if 0 <= test_task_idx[i] <= 14:
                    outputs = list(batch['output'][0])
                    dist = image_similarity(predictions, outputs, vit, vit_extractor)
                    task_rewards.append(dist/100)
                elif 15 <= test_task_idx[i] <= 104 or 107 <= test_task_idx[i] <= 184:
                    outputs = list(batch['output'][0])
                    f1 = np.mean(txt_eval(predictions, outputs, bertscore, device="cuda:0"))
                    task_rewards.append(f1)
                else:
                    predictions = [pred for pred in predictions]
                    inputs = [text for text in inputs[0]]
                    score = clip_score(predictions, inputs)
                    task_rewards.append(score.detach()/100)
            ave_task_reward = np.mean(task_rewards)
            logging.info("Average reward on current task: " + str(ave_task_reward))
            seqCombination.close_module_seq()
            
        rewards.append(ave_task_reward)
        if 0<=test_task_idx[i]<=14:
            similairies.append(ave_task_reward)
        elif 15<=test_task_idx[i]<=104 or 107<=test_task_idx[i]<=184:
            berts.append(ave_task_reward)
        else:
            clips.append(ave_task_reward)

logging.info("Finished testing!")    
logging.info(f'Clips: {np.mean(clips)}, BERTS: {np.mean(berts)}, ViT: {np.mean(similairies)}, Rewards: {np.mean(rewards)}')
print(test_task_idx, berts, np.mean(berts[:-3]), np.mean(rewards[:-3]), np.mean(berts[-3:]))
