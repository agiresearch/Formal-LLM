from sentence_transformers import SentenceTransformer, util
import os
import sys
sys.path.append('./')
print(sys.path)
from general_dataset import GeneralDataset
from torch.utils.data import DataLoader
import torch
from agi_utils import *
import torch
import numpy as np
from IPython.utils import io
from agi_utils import *
import random
from tqdm import tqdm
from evaluate import load
from torchvision import transforms
from transformers import AutoModel, AutoFeatureExtractor
from torchmetrics.multimodal import CLIPScore
from combine_model_seq import SeqCombine

data_path = "./openagi_data/"

task_discriptions = txt_loader(data_path + "task_description.txt")
test_task_idx = [2, 3, 10, 15, 20, 35, 45, 55, 65, 70, 90, 106, 112, 115, 118, 177, 179]
test_dataloaders = []
for i in test_task_idx:
    dataset = GeneralDataset(i, data_path)
    dataloader = DataLoader(dataset, batch_size=5)
    test_dataloaders.append(dataloader)

test_tasks = [task_discriptions[i].strip() for i in test_task_idx]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_name", type=str, default='./0.txt')
parser.add_argument("--huggingface_cache", type=str, default='./')

args = parser.parse_args("")

# device for bert score
eval_device = "cuda:3"
args.device_list = ["cuda:0","cuda:1","cuda:2","cpu"]
clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

# Load a pre-trained Vision Transformer model and its feature extractor
vit_ckpt = "nateraw/vit-base-beans"
vit = AutoModel.from_pretrained(vit_ckpt)
vit.eval()
vit_extractor = AutoFeatureExtractor.from_pretrained(vit_ckpt)

f = transforms.ToPILImage()
bertscore = load("bertscore")

sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

seqCombination = SeqCombine(args)

from openai import OpenAI
from copy import deepcopy
import logging

def get_valid_response(max_num):
    temperature = 0
    while True:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            max_tokens=25,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["."]
        )

        response = response.choices[0].message.content
        response.isdigit()
        logging.info(response)
        if response.isdigit() and 1 <= int(response) <= max_num:
            break
        temperature += 0.1
        if temperature > 5:
            exit()
    return int(response)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename=args.log_name, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

api_key = "YOUR OPENAI TOKEN"
client = OpenAI(api_key=api_key)

from generate_model_seq import SeqGen
seqGen = SeqGen(None, None, None)

task2type = dict()
for index in range(0, 6):
    for task in seqGen.candidates[index]['task_list']:
        task2type[task] = index

rewards = []
clips = []
berts = []
similairies = []
messages = []

for i, task_description in enumerate(test_tasks):
    messages = []
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
            prompt = f'You will help me generate a plan for the problem: "{task_description}" by answering a series of my questions.\nQuestion:```'

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
                available = [1, 2, 4, 5]  # i2t, t2t, tt2t, it2t
                if 'image' not in test_dataloaders[i].dataset.input_file_type and len(necessary_t2i) <= auto_stack.count('I'):
                    available = [2, 4]  # remove i2t, it2t
                if 'text' not in test_dataloaders[i].dataset.input_file_type and len(necessary_i2t) < 2 + auto_stack.count('T'):
                    available.remove(4)  # remove tt2t
                for index in available:
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0 and 'text' in test_dataloaders[i].dataset.input_file_type:
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
                available = [0, 3]  # i2i, t2i
                if 'text' not in test_dataloaders[i].dataset.input_file_type and len(necessary_i2t) <= auto_stack.count('T'):
                    available = [0]  # remove t2i
                for index in available:
                    for task in seqGen.candidates[index]['task_list']:
                        if task[:-1] not in action:
                            tool_cnt += 1
                            model_choice.append(task)
                            prompt += f'{tool_cnt}: the output of {task}\n'
                if len(action) > 0 and 'image' in test_dataloaders[i].dataset.input_file_type:
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

            prompt += f"```\nYour answer should be only an number, referring to the desired choice. Don't be verbose!"
            logging.info(prompt)

            messages.append({
                "role": "user",
                "content": prompt,
            })

            choice = get_valid_response(tool_cnt)
            logging.info(choice)

            messages.append({
                "role": "assistant",
                "content": str(choice),
            })

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
                elif index == 5:  # it2t
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
                    task_rewards.append(dist / 100)
                elif 15 <= test_task_idx[i] <= 104 or 107 <= test_task_idx[i] <= 184:
                    outputs = list(batch['output'][0])
                    f1 = np.mean(txt_eval(predictions, outputs, bertscore, device="cuda:0"))
                    task_rewards.append(f1)
                else:
                    predictions = [pred for pred in predictions]
                    inputs = [text for text in inputs[0]]
                    score = clip_score(predictions, inputs)
                    task_rewards.append(score.detach() / 100)
            ave_task_reward = np.mean(task_rewards)
            logging.info("Average reward on current task: " + str(ave_task_reward))
            seqCombination.close_module_seq()

        rewards.append(ave_task_reward)
        if 0 <= test_task_idx[i] <= 14:
            similairies.append(ave_task_reward)
        elif 15 <= test_task_idx[i] <= 104 or 107 <= test_task_idx[i] <= 184:
            berts.append(ave_task_reward)
        else:
            clips.append(ave_task_reward)

logging.info("Finished testing!")
logging.info(f'Clips: {np.mean(clips)}, BERTS: {np.mean(berts[:-4])}, ViT: {np.mean(similairies)}, Difficult: {np.mean(berts[-4:])}, Rewards: {np.mean(rewards)}')
logging.info('All BERTS: ' + str(berts))
