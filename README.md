# Word2World

![image](https://github.com/umair-nasir14/Word2World/assets/68095790/c7e5af2e-a948-4eda-9e9c-4c0e0f0f2f46)

This repository contains to code for Word2World: Generating Stories and Worlds through Large Language Models.

### Abstract:

Large Language Models (LLMs) have proven their worth across a diverse spectrum of disciplines. LLMs have shown great potential in Procedural Content Generation (PCG) as well, but directly generating a level through a pre-trained LLM is still challenging. This work introduces \texttt{Word2World}, a system that enables LLMs to procedurally design playable games through stories, without any task-specific fine-tuning. \texttt{Word2World} leverages the abilities of LLMs to create diverse content and extract information. Combining these abilities, LLMs can create a story for the game, design narrative, and place tiles in appropriate places to create coherent worlds and playable games. We test \texttt{Word2World} with different LLMs and perform a thorough ablation study to validate each step.

### Usage:

Clone the repo:

`https://github.com/umair-nasir14/Word2World.git`

Install the environment and activate it:

```
cd Word2World
conda env create -f environment.yml
conda activate Word2World
```

Run with default configs:

`python main.py`

Or run with specified configs:

```
python main.py \
--model="gpt-4-turbo-2024-04-09" \
--min_story_paragraphs=4 \
--max_story_paragraphs=5 \
--total_objectives=8 \
--rounds=3 \
--experiment_name="Your_Word2World" \
--save_dir="outputs"
```

### Results:

#### LLM comparison:

![image](https://github.com/umair-nasir14/Word2World/assets/68095790/7b843e04-d009-4708-9b3e-686ddfe9c358)

#### Worlds:

![world_1](https://github.com/umair-nasir14/Word2World/assets/68095790/5b85bb03-eed4-4879-ab07-4683d317ab20)
![world_2](https://github.com/umair-nasir14/Word2World/assets/68095790/6ccaa7e3-6573-4f20-b3a9-03e8992ffc9c)
![world_3](https://github.com/umair-nasir14/Word2World/assets/68095790/53e38643-d10a-4c16-a584-c0aa19116e60)
![world_4](https://github.com/umair-nasir14/Word2World/assets/68095790/fc8df4a5-63db-414f-96ca-a4094397ff9d)
![world_6](https://github.com/umair-nasir14/Word2World/assets/68095790/d92fa869-82de-4e97-bb77-2eb5fb7d04e2)
![world_7](https://github.com/umair-nasir14/Word2World/assets/68095790/751a753e-9e3d-41da-b146-fa852d0e7f1c)

### Cite:
```

```

