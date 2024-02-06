# Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents

<a href='https://arxiv.org/abs/2402.00798'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> 

Recent advancements on Large Language Models (LLMs) enable AI Agents to automatically generate and execute multi-step plans to solve complex tasks. However, since LLM's content generation process is hardly controllable, current LLM-based agents frequently generate invalid or non-executable plans, which jeopardizes the performance of the generated plans and corrupts users' trust in LLM-based agents. In response, this work proposes a novel "Formal-LLM" framework for LLM-based agents by integrating the expressiveness of natural language and the precision of formal language. Specifically, the framework allows human users to express their requirements or constraints for the planning process as an automaton. A stack-based LLM plan generation process is then conducted under the supervision of the automaton to ensure that the generated plan satisfies the constraints, making the planning process controllable. We conduct experiments on both benchmark tasks and practical real-life tasks, and our framework achieves over 50% overall performance increase, which validates the feasibility and effectiveness of employing Formal-LLM to guide the plan generation of agents, preventing the agents from generating invalid and unsuccessful plans. Further, more controllable LLM-based agents can facilitate the broader utilization of LLM in application scenarios where high validity of planning is essential.

<p align="center">
<img src="images/Formal-LLM.png">
</p>


## Citation

```
@article{li2024formalllm,
      title={Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents}, 
      author={Zelong Li and Wenyue Hua and Hao Wang and He Zhu and Yongfeng Zhang},
      journal={arXiv:2402.00798},
      year={2024}
}
```

## Requirements

- Python==3.9
- PyTorch==1.13.1
- transformers==4.28.0
- langchain==0.0.153
- peft==0.7.1

## Usage

0. Clone this repo.

1. Create a conda virtual environment and install the Pytorch matching your CUDA version. For example, for CUDA version 11.7:

```
conda create -n your_env_name python=3.9
conda activate your_env_name

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

2. Install necessary packages:

```
pip install -r requirements.txt
```

3. Download the preprocessed data from this [Google Drive link](https://drive.google.com/drive/folders/1AjT6y7qLIMxcmHhUBG5IE1_5SnCPR57e?usp=share_link), put it into the *FormalLLM/src/* folder, then unzip it.

4. The codes for the real-world tasks are in the *Formal/src/realworld_tasks* folder, and the codes for benchmark tasks are in the *Formal/src/open_source* and *Formal/src/closed_source* folders. Specifically, the *open_source* LLMs include Flan-T5-large, Vicuna-7B, and LLaMA-2-13B. The *closed_source* LLMs include GPT-3.5-turbo, Claude-2, and GPT-4.

5. Before running any codes, you need replace "YOUR OPENAI TOKEN", "HUGGINGFACE TOKEN", and "YOUR CLAUDE TOKEN", with your OpenAI, Huggingface, and Claude API keys, respectively.

6. Make sure you are in the *Formal/src* folder before running the codes. Otherwise,

```
cd src
```

7. To evaluate the performance of *closed_source* LLM-based agent with our Formal-LLM framework, you need to run Python with these arguments: *huggingface_cache* and *log_name*. For example,

```
python closed_source/FLLM_ChatGPT.py --log_name ./0.txt --huggingface_cache ./
```

8. To evaluate the performance of *open_source* LLM-based agent with our Formal-LLM framework, you need to run Python with these arguments: *cache_dir*, *log_name*, and *ckp_dir* (directory for saving checkpoints). For example,

```
python open_source/FLLM_schema-flan-t5.py --log_name ./0.txt --cache_dir ./ --ckp_dir ./
```

9. To evaluate the performance on real-life tasks (use daily planning as an example):

```
python realworld_tasks/daily.py
```

## Reference

- We leveraged the dataset of [OpenAGI](https://github.com/agiresearch/OpenAGI) projects to implement our experiment.
