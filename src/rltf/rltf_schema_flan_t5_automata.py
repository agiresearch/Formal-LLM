#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--num_seq", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--epsilon", type=float, default=1.0)
parser.add_argument("--decay_rate", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-6)
parser.add_argument("--accumulate_steps", type=int, default=1)
parser.add_argument("--warm_up_proportion", type=float, default=0.1)
parser.add_argument('--bl_dec', type=float, default=0.8)
parser.add_argument('--pos_reward_coef', type=float, default=1.0)
parser.add_argument("--cache_dir", type=str, default="/common/users/zl359/cache_dir")
parser.add_argument("--log_name", type=str, default='/common/home/zl359/OpenAGI/automata_log/3.txt')
parser.add_argument("--ckp_dir", type=str, default='/common/home/zl359/OpenAGI/checkpoint/flan_t5_base_automata_5/')

args = parser.parse_args("")


# In[3]:


"""
load training and test datasets
"""
data_path = "./openagi_data/"

task_discriptions = txt_loader("./task_description.txt")
training_task_idx = [7,20,30,40,50,60, 0, 105, 114, 112, 182, 118, 119]
test_task_idx = [2,3,10,15,20,35,45,55,65,70,90,106,107, 177, 171, 116]
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


# In[4]:


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


# In[5]:


from undecorated import undecorated
from finetune.utils import construct_optimizer
from types import MethodType
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

device = 'cuda:0'

tokenizer = T5Tokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
backbone_model = T5ForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir, device_map='auto')#'balanced_low_0')# .to(device)
# backbone_model.load_state_dict(torch.load("./finetune/10_shot_finetuned.pt", map_location="cpu"))
# backbone_model = T5ForConditionalGeneration.from_pretrained('/common/home/zl359/OpenAGI/checkpoint/flan_t5_base_automata/', cache_dir=args.cache_dir)
# device_map = infer_auto_device_map(backbone_model, max_memory={0: "1000MiB", 1: "1000MiB", 2: "1000MiB", "cpu": "30GiB"})
# backbone_model = load_checkpoint_and_dispatch(backbone_model, checkpoint="./finetune/10_shot_finetuned.pt", device_map=device_map)
# backbone_model = load_checkpoint_and_dispatch(backbone_model, '/common/home/zl359/OpenAGI/checkpoint/flan_t5_base_automata/', device_map='auto')
# backbone_model = backbone_model.to(device)

seqGen = SeqGen(backbone_model, tokenizer, device)

generate_with_grad = undecorated(seqGen.model.generate)
seqGen.model.generate_with_grad = MethodType(generate_with_grad, seqGen.model)
# optimizer = optim.SGD(seqGen.model.parameters(), lr=0.0001, momentum=0.9)
optimizer, scheduler = construct_optimizer(args, seqGen.model, 20)


# In[6]:


import logging
# args.log_name = '/common/home/zl359/OpenAGI/automata_log/2.txt'
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args.log_name, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(vars(args))


# ## Baseline

# In[7]:


task2type = dict()
for index in range(0, 6):
    for task in seqGen.candidates[index]['task_list']:
        task2type[task] = index
baseline = [] #[0] * len(training_tasks)

for i, task_description in enumerate(training_tasks):
    with torch.no_grad():
        logging.info(f'Solving tasks: {i + 1} / {len(training_tasks)}')
        # variable_stack = []
        # variable_mapping = dict()
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
            # if len(variable_stack
            # variable_index = variable_stack
            
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
            
            #choice = output_sequence[0]
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
    
            # random_num = random.random()
            choice = torch.argmax(torch.stack(log_probs).detach())
            choice = choice.item() + 1
            logging.info(choice)
            
            # try:
            #     choice = int(output_sequence[0])
            # except:
            #     print('invalid output')
            #     auto_stack.append(choice)
            #     break
            if choice <= 0 or choice > tool_cnt:
                # print('Answer Out of Range')
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
            # logging.info()
        
        module_list = action[::-1]
        logging.info(f'Module Sequence: {module_list}')
        
        if len(auto_stack) > 0:
            # print('Invalid or Too long plan')
            logging.info('Invalid or Too long plan')
            ave_task_reward = 0
        else:
            seqCombination.construct_module_tree(module_list)
            task_rewards = []
            for idx, batch in enumerate(tqdm(training_dataloaders[i])):
                inputs = [list(input_data) for input_data in batch['input']]
                # seqCombination.construct_module_tree(module_list)
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
            # print("Average reward on current task: " + str(ave_task_reward))
            logging.info("Average reward on current task: " + str(ave_task_reward))
            seqCombination.close_module_seq()
            
        baseline.append(ave_task_reward)

print(np.mean(baseline))


# ## Training

# In[9]:


epochs = args.epochs
epsilon = args.epsilon
decay_rate = args.decay_rate
random.seed(args.seed)

task2type = dict()
for index in range(0, 6):
    for task in seqGen.candidates[index]['task_list']:
        task2type[task] = index

# baseline = [0] * len(training_tasks)

for e in range(epochs):
    epoch_total_loss = 0.0
    rewards = []
    # print('num of epoch ' + str(e+1))
    logging.info(f'num of epoch {e + 1}')
    optimizer.zero_grad()
    for i, task_description in enumerate(training_tasks):
        total_loss = 0
        total_log_prob = 0
        # print(f'Solving tasks: {i + 1} / {len(training_tasks)}')
        logging.info(f'Solving tasks: {i + 1} / {len(training_tasks)}')
        # variable_stack = []
        # variable_mapping = dict()
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
            # if len(variable_stack
            # variable_index = variable_stack
            
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
            logits = torch.stack(log_probs)#.detach()
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
            if random_num >= epsilon:
                choice = torch.argmax(logits)
            else:
                choice = torch.distributions.Categorical(logits=logits).sample() 
            total_log_prob -= logits[choice]
            choice = choice.item() + 1
            logging.info(choice)
            
            # try:
            #     choice = int(output_sequence[0])
            # except:
            #     print('invalid output')
            #     break
            if choice <= 0 or choice > tool_cnt:
                # print('Answer Out of Range')
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
            # logging.info()
        
        module_list = action[::-1]
        logging.info(f'Module Sequence: {module_list}')
        
        if len(auto_stack) > 0:
            # print('Invalid or Too long plan')
            logging.info('Invalid or Too long plan')
        else:
            with torch.no_grad():
                seqCombination.construct_module_tree(module_list)
                task_rewards = []
                for idx, batch in enumerate(tqdm(training_dataloaders[i])):
                    inputs = [list(input_data) for input_data in batch['input']]
                    # seqCombination.construct_module_tree(module_list)
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
                # print("Average reward on current task: " + str(ave_task_reward))
                logging.info("Average reward on current task: " + str(ave_task_reward))
                rewards.append(ave_task_reward)
                seqCombination.close_module_seq()
        total_loss += total_log_prob * (ave_task_reward - baseline[i])
        if total_loss > 0.0:
            total_loss *= args.pos_reward_coef
        total_loss.backward()
        # baseline[i] = ave_task_reward
        # baseline[i] = (baseline[i] * e + ave_task_reward) / (e + 1)
        baseline[i] -= (1 - args.bl_dec) * (baseline[i] - ave_task_reward)
        epoch_total_loss += total_loss.detach()
        
    avg_reward = np.mean(rewards)
    # print(f"Average reward: {avg_reward}")
    logging.info(f"Average reward: {avg_reward}")
    # total_loss *= (avg_reward - baseline)
    # print(f"Loss: {total_loss.item()}")
    logging.info(f"Loss: {epoch_total_loss.item()}")
    
    optimizer.step()
    scheduler.step()
    epsilon *= decay_rate
    # baseline = avg_reward


# In[10]:


seqGen.model.save_pretrained(args.ckp_dir)


# In[ ]:


prompt = 'Who is Obama?'
input_ids = seqGen.tokenizer.batch_encode_plus([prompt], padding="longest", return_tensors="pt")["input_ids"]
input_ids = input_ids.to(device)
output = seqGen.model.generate_with_grad(input_ids, max_length=50, min_length=1, return_dict_in_generate=True, output_scores=True, 
                                         output_hidden_states=True, renormalize_logits=True, temperature=1, top_k=5, top_p=0.5,
                                         num_return_sequences=1)
output_ids = output["sequences"][:, 1:]
output_sequence = [s.replace("<pad>", "").replace("</s>", "") for s in seqGen.tokenizer.batch_decode(output_ids)]
print(output_sequence)


# ## Testing

# In[ ]:


# test_task_idx = [2,3,10,15,20,35,45,55,65,70,90,106,107]
# test_dataloaders = []
    
# for j in test_task_idx:
#     dataset = GeneralDataset(j,data_path)
#     dataloader = DataLoader(dataset, batch_size=args.batch_size)
#     test_dataloaders.append(dataloader)
    
# test_tasks = [task_discriptions[j].strip() for j in test_task_idx]


# In[11]:


# logging.info('Free Form Generation')
logging.info('Generation with Automata as constrain')

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
        # variable_stack = []
        # variable_mapping = dict()
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
            # if len(variable_stack
            # variable_index = variable_stack
            
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
            # choice = output_sequence[0]
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
    
            # random_num = random.random()
            choice = torch.argmax(torch.stack(log_probs).detach())
            choice = choice.item() + 1
            logging.info(choice)
            
            # try:
            #     choice = int(output_sequence[0])
            # except:
            #     print('invalid output')
            #     auto_stack.append(choice)
            #     break
            if choice <= 0 or choice > tool_cnt:
                # print('Answer Out of Range')
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
            # logging.info()
        
        module_list = action[::-1]
        logging.info(f'Module Sequence: {module_list}')
        
        if len(auto_stack) > 0:
            # print('Invalid or Too long plan')
            logging.info('Invalid or Too long plan')
            ave_task_reward = 0
        else:
            seqCombination.construct_module_tree(module_list)
            task_rewards = []
            for idx, batch in enumerate(tqdm(test_dataloaders[i])):
                inputs = [list(input_data) for input_data in batch['input']]
                # seqCombination.construct_module_tree(module_list)
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
            # print("Average reward on current task: " + str(ave_task_reward))
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


# In[13]:


test_task_idx, berts, np.mean(berts[:-3]), np.mean(rewards[:-3])


# In[ ]:


# # seqCombination.close_module_seq()
# log_probs = []
# for t in range(1, tool_cnt + 1):
#     out_id = torch.tensor([seqGen.tokenizer.encode(str(t))])
#     print(out_id)
#     logprob = 0
#     length = out_id.size(-1)
#     for l in range(length):
#         score = scores[l][0]
#         print(score)
#         if seqGen.tokenizer.decode(out_id[0][l]) == "</s>":
#             continue
#         logprob += score[out_id[0][l]]
#         print(out_id[0][l], score[out_id[0][l]])
#     log_probs.append(logprob)
# print(log_probs)


# In[ ]:


# output = seqCombination.image_colorization(inputs[0], seqCombination.auto_used_device_list['Colorization'])
# output = seqCombination.image_denoising(output, seqCombination.auto_used_device_list['Image Denoising'])
# output = seqCombination.image_deblurring(output, seqCombination.auto_used_device_list['Image Deblurring'])
# output = seqCombination.image_object_detect(output, seqCombination.auto_used_device_list['Object Detection'])
# output = seqCombination.fill_mask(output, seqCombination.auto_used_device_list['Fill Mask'])
# print(output)
# output = seqCombination.sentiment_analysis(output, seqCombination.auto_used_device_list['Sentiment Analysis'])
# print(output)
# # output = seqCombination.summarization_tokenizer(output, return_tensors="pt", padding=True).to(seqCombination.auto_used_device_list['Text Summarization'])
# # with torch.no_grad():
# #     output = seqCombination.summarizer.generate(**output)
# # output = [seqCombination.summarization_tokenizer.decode(summary_ids, skip_special_tokens=True).strip("</s>") for summary_ids in output]
# output = seqCombination.text_summarization(output, seqCombination.auto_used_device_list['Text Summarization'])
# # output = seqCombination.vqa([output, inputs[0]], seqCombination.auto_used_device_list['Visual Question Answering'])
# # print(output)
# encoding = seqCombination.vqa_processor(inputs[0], output, return_tensors="pt", padding=True, truncation=True).to(seqCombination.auto_used_device_list['Visual Question Answering'])
# print(encoding['input_ids'].shape)
# with torch.no_grad():
#     output = seqCombination.vqa_model(**encoding)
# output
# with torch.no_grad():
#     output_3 = seqCombination.text_generator.generate(**output_2, min_length=5, max_new_tokens=30)
# output_3


# In[ ]:


# epochs = args.epochs
# epsilon = args.epsilon
# decay_rate = args.decay_rate
# random.seed(0)


# baseline = 0
# rewards = []
# for i, task_description in enumerate(training_tasks):        
#     auto_stack = []
#     action = []
#     if training_dataloaders[i].dataset.output_file_type[0] == 'text':
#         auto_stack.append('T')
#     elif training_dataloaders[i].dataset.output_file_type[0] == 'image':
#         auto_stack.append('I')
#     else:
#         raise NotImplementedError
    
#     while len(action) < 5 and len(auto_stack) > 0:
#         stack_top = auto_stack.pop()
#         tool_cnt = 0
#         prefix_sum = [0]
#         if stack_top == 'T':
#             prompt = f'You will help me generate a plan for the Problem: "{task_description}" by answering a series of my questions.\n'
#             if len(action) == 0:
#                 prompt += f'To get the final text, we have the following choices:\n'
#             else:
#                 prompt += f'To get the output text of "{action[-1]}", we have the following choices:\n'
#             for task in seqGen.candidates[1]['task_list']: # i2t
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: the output of {task}\n'
#             prefix_sum.append(tool_cnt)
#             for task in seqGen.candidates[2]['task_list']: # t2t
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: the output of {task}\n'
#             prefix_sum.append(tool_cnt)
#             for task in seqGen.candidates[4]['task_list']: # tt2t
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: {task}\n'
#             prefix_sum.append(tool_cnt)
#             for task in seqGen.candidates[5]['task_list']: # it2t
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: the output of {task}\n'
#             prefix_sum.append(tool_cnt)
#             if 'text' in training_dataloaders[i].dataset.input_file_type:
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: Input Text.\n'
#                 prefix_sum.append(tool_cnt)
            
#             prompt += '\nCurrect Progress: \n'
#             for j, step in enumerate(action):
#                 if j > 0:
#                     prompt += f'Step (n-{j}): Use {step}\n'
#                 else:
#                     prompt += f'Step n: Use {step}\n'
#             if len(action) == 0:
#                 prompt += 'Step n: ?\n\n'
#             else:
#                 prompt += f'Step (n-{len(action)}): ?\n\n'
#             prompt += f'Considering the whole task is to: "{task_description}", select the one to get the text. '
#             prompt += f'Your answer should be only an integer, referring to desired choice.\n' 
#             print(prompt)
            
#         elif stack_top == 'I':
#             prompt = f'You will help me generate a plan for the Problem: "{task_description}" by answering a series of my questions.\n'
#             if len(action) == 0:
#                 prompt += f'To get the final image, we have the following choices:\n'
#             else:
#                 prompt += f'To get the output image of "{action[-1]}", we have the following choices:\n'
#             for task in seqGen.candidates[0]['task_list']: # i2i
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: the output of {task}\n'
#             prefix_sum.append(tool_cnt)
#             for task in seqGen.candidates[3]['task_list']: # t2i
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: the output of {task}\n'
#             prefix_sum.append(tool_cnt)
#             if 'image' in training_dataloaders[i].dataset.input_file_type:
#                 tool_cnt += 1
#                 prompt += f'{tool_cnt}: Input Image.\n'
#                 prefix_sum.append(tool_cnt)

#             prompt += '\nCurrect Progress: \n'
#             for j, step in enumerate(action):
#                 if j > 0:
#                     prompt += f'Step (n-{j}): Use {step}\n'
#                 else:
#                     prompt += f'Step n: Use {step}\n'
#             if len(action) == 0:
#                 prompt += 'Step n: ?\n\n'
#             else:
#                 prompt += f'Step (n-{len(action)}): ?\n\n'
#             prompt += f'Considering the whole task is to: "{task_description}", select the one to get the text.'
#             prompt += f'Your answer should be only an integer, referring to desired choice.\n' 
#             print(prompt)
#         else:
#             raise NotImplementedError
        
#         input_ids = seqGen.tokenizer.batch_encode_plus([prompt], padding="longest", return_tensors="pt")["input_ids"]
#         input_ids = input_ids.to(device)
#         output = seqGen.model.generate_with_grad(input_ids, max_length=3, min_length=1, return_dict_in_generate=True, output_scores=True, 
#                                                  output_hidden_states=True, renormalize_logits=True, temperature=1, top_k=5, top_p=0.5,
#                                                  num_return_sequences=1)
#         output_ids = output["sequences"][:,1:]
#         output_sequence = [s.replace("<pad>", "").replace("</s>", "") for s in seqGen.tokenizer.batch_decode(output_ids)]
#         scores = output['scores']
#         log_probs = []
#         for t in range(1, tool_cnt + 1):
#             out_id = torch.tensor([seqGen.tokenizer.encode(str(t))])
#             logprob = 0
#             length = out_id.size(-1)
#             for l in range(length):
#                 score = scores[l][0]
#                 if seqGen.tokenizer.decode(out_id[0][l]) == "</s>":
#                     continue
#                 logprob += score[out_id[0][l]]
#             loss = logprob
#             log_probs.append(loss)
#         print(log_probs)

#         random_num = random.random()
#         if random_num >= epsilon:
#             actions = torch.argmax(torch.stack(log_probs).detach()) + 1
#         else:
#             actions = torch.distributions.Categorical(torch.stack(log_probs).detach()).sample() + 1
#         choice = actions.item()
#         print(choice)
        
#         # try:
#         #     choice = int(output_sequence[0])
#         # except:
#         #     print('invalid output')
#         #     break
#         if choice < 0 or choice > tool_cnt:
#             print('Answer Out of Range')
#             break
        
#         if stack_top == 'T':
#             for j, index in enumerate([1, 2, 4, 5]):
#                 if choice <= prefix_sum[j + 1]:
#                     if index == 1:  # i2t
#                         auto_stack.append('I')
#                     elif index == 2:  # t2t
#                         auto_stack.append('T')
#                     elif index == 4:  # tt2t
#                         auto_stack.append('TT')
#                     elif index == 5: # it2t
#                         auto_stack.append('IT')
#                     else:
#                         raise NotImplementedError
#                     action.append(seqGen.candidates[index]['task_list'][choice - prefix_sum[j] - 1])
#                     break
#                 elif index == 5: # use Input
#                     action.append('Input Text')
#         elif stack_top == 'I':
#             for j, index in enumerate([0, 3]):
#                 if choice <= prefix_sum[j + 1]:
#                     if index == 0:  # i2i
#                         auto_stack.append('I')
#                     elif index == 3:  # t2i
#                         auto_stack.append('T')
#                     else:
#                         raise NotImplementedError
#                     action.append(seqGen.candidates[index]['task_list'][choice - prefix_sum[j] - 1])
#                     break
#                 elif index == 3: # use Input
#                     action.append('Input Image')
#         else:
#             raise NotImplementedError
#         print()
#     module_list = action[::-1]
#     print(f'Module Sequence: {module_list}')
    
#     if len(auto_stack) > 0:
#         print('Invalid or Too long plan')
#     else:
#         seqCombination.construct_module_seq(module_list)
#         for idx, batch in enumerate(tqdm(training_dataloaders[i])):
#             inputs = list(batch['input'][0])
#             seqCombination.construct_module_seq(module_list)
#             predictions = seqCombination.run_module_seq(inputs)

#             if 0<=training_task_idx[i]<=14:
#                 outputs = list(batch['output'][0])
#                 dist = image_similarity(predictions, outputs, vit, vit_extractor)
#                 task_rewards.append(dist/100)
#             elif 15<=training_task_idx[i]<=104 or 107<=task_idx[i]<=184:
#                 outputs = list(batch['output'][0])
#                 f1 = np.mean(txt_eval(predictions, outputs, bertscore, device="cuda:4"))
#                 task_rewards.append(f1)
#             else:
#                 score = clip_score(predictions, inputs)
#                 task_rewards.append(score.detach()/100)
#         ave_task_reward = np.mean(task_rewards)    
#         print("Average reward on current task: " + str(ave_task_reward))
#         rewards.append(ave_task_reward)
#         seqCombination.close_module_seq()


# ### Training

# In[ ]:


# epochs = args.epochs
# epsilon = args.epsilon
# decay_rate = args.decay_rate


# for e in range(epochs):
#     baseline = 0
#     rewards = []
    
#     print('num of epoch ' + str(e+1))
#     for i, task_description in enumerate(training_tasks):
#         task_rewards = []
#         # print(task_description)
#         optimizer.zero_grad()
#         generated_module_seq, log_prob = seqGen.generate_sequence([training_tasks[i]],\
#                                                                    module_length=10, \
#                                                                    beam_size=30, \
#                                                                    num_seq=30,\
#                                                                    top_k=5,\
#                                                                    top_p=0.5,\
#                                                                    temperature=0.9,\
#                                                                    constraint=[0,100],\
#                                                                    num_beam_groups=1)

#         if random.random() >= epsilon:
#             action = torch.argmax(torch.stack(log_prob).detach())
#         else:
#             action = torch.distributions.Categorical(torch.stack(log_prob).detach()).sample()

#         # decrease epsilon by the decay rate after each step
#         epsilon *= decay_rate

#         module_list = generated_module_seq[action][:-1]

#         if module_seq_filter(module_list, training_task_idx[i]):

#             # print("Module Sequence: " + module_list)
#             seqCombination.construct_module_seq(module_list)


#             for idx, batch in enumerate(tqdm(training_dataloaders[i])):
#                 inputs = list(batch['input'][0])
#                 seqCombination.construct_module_seq(module_list)
#                 predictions = seqCombination.run_module_seq(inputs)

#                 if 0<=training_task_idx[i]<=14:
#                     outputs = list(batch['output'][0])
#                     dist = image_similarity(predictions, outputs, vit, vit_extractor)
#                     task_rewards.append(dist/100)
#                 elif 15<=training_task_idx[i]<=104 or 107<=task_idx[i]<=184:
#                     outputs = list(batch['output'][0])
#                     f1 = np.mean(txt_eval(predictions, outputs, bertscore, device="cuda:4"))
#                     task_rewards.append(f1)
#                 else:
#                     clip_score = score = clip_score(predictions, inputs)
#                     task_rewards.append(clip_score.detach()/100)
#             ave_task_reward = np.mean(task_rewards)    
#             # print("Average reward on current task: " + str(ave_task_reward))
#             rewards.append(ave_task_reward)

#             seqCombination.close_module_seq()
#         else:
#             rewards.append(-1)
            
#         avg_reward = np.mean(rewards)
#         print("Average reward: " + str(avg_reward))
#         loss = -log_prob[action] * (avg_reward - baseline)
#         print("Loss: "+ str(loss.item()))
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         # baseline = avg_reward

# print("Finished training!")    


# ### Test

# In[ ]:


# rewards = []
# clips = []
# berts = []
# similairies = []

# for i, task_description in enumerate(test_tasks):
#     task_rewards = []
#     with torch.no_grad():
#         generated_module_seq, log_prob = seqGen.generate_sequence([test_tasks[i]],\
#                                                                    module_length=10, \
#                                                                    beam_size=30, \
#                                                                    num_seq=30,\
#                                                                    top_k=5,\
#                                                                    top_p=0.5,\
#                                                                    temperature=0.9,\
#                                                                    constraint=[0,100],\
#                                                                    num_beam_groups=1)

#     action = torch.argmax(torch.stack(log_prob).detach())
    

#     module_list = generated_module_seq[action][:-1]
#     # print(task_description)
#     # print("Module Sequence: " + module_list)

#     if module_seq_filter(module_list, test_task_idx[i]):
#         seqCombination.construct_module_seq(module_list)

#         for idx, batch in enumerate(tqdm(test_dataloaders[i])):
#             inputs = list(batch['input'][0])
#             predictions = seqCombination.run_module_seq(inputs)

#             if 0<=test_task_idx[i]<=14:
#                 outputs = list(batch['output'][0])
#                 dist = image_similarity(predictions, outputs, vit, vit_extractor)
#                 task_rewards.append(dist/100)
#             elif 15<=test_task_idx[i]<=104 or 107<=test_task_idx[i]:
#                 outputs = list(batch['output'][0])
#                 f1 = np.mean(txt_eval(predictions, outputs, bertscore))
                
#                 task_rewards.append(f1)
#             else:
#                 score = clip_score(predictions, inputs)
#                 task_rewards.append(score.detach()/100)
                
#         ave_task_reward = np.mean(task_rewards)    
        
        
#         seqCombination.close_module_seq()
            
#     else:
#         ave_task_reward = 0
        
#     if 0<=test_task_idx[i]<=14:
#         similairies.append(ave_task_reward)
#     elif 15<=test_task_idx[i]<=104 or 107<=test_task_idx[i]<=184:
#         berts.append(ave_task_reward)
#     else:
#         clips.append(ave_task_reward)

#     rewards.append(ave_task_reward)     
    

# print("Finished testing!")    


# In[ ]:


# np.mean(clips), np.mean(berts), np.mean(similairies), np.mean(rewards)


# In[ ]:




