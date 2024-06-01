# Word2World

![image](https://github.com/umair-nasir14/Word2World/assets/68095790/c7e5af2e-a948-4eda-9e9c-4c0e0f0f2f46)

This repository contains to code for [Word2World: Generating Stories and Worlds through Large Language Models](https://arxiv.org/abs/2405.06686).

### Abstract:

Large Language Models (LLMs) have proven their worth across a diverse spectrum of disciplines. LLMs have shown great potential in Procedural Content Generation (PCG) as well, but directly generating a level through a pre-trained LLM is still challenging. This work introduces `Word2World`, a system that enables LLMs to procedurally design playable games through stories, without any task-specific fine-tuning. `Word2World` leverages the abilities of LLMs to create diverse content and extract information. Combining these abilities, LLMs can create a story for the game, design narrative, and place tiles in appropriate places to create coherent worlds and playable games. We test `Word2World` with different LLMs and perform a thorough ablation study to validate each step.

### Usage:

Clone the repo:

`https://github.com/umair-nasir14/Word2World.git`

Install the environment and activate it:

```
cd Word2World
type > word2world/.env
conda env create -f environment.yml
conda activate word2world
```

Add your API key to the .env file created in word2world folder:

```
OPENAI_API_KEY="sk..."
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
--rounds=1 \
--experiment_name="Your_World" \
--save_dir="outputs"
```

To play the generated game:

```
python word2world/play_game.py "path_to_game_data\game_data.json"
```
where `game_data.json` is generated when the Word2World loop is finished and is saved to `\outputs\game_data.json`. This can be modified in `configs` or as `--save_dir` arg.

To play an example world:

```
python word2world/play_game.py
```

### Results:

#### LLM comparison:

![image](https://github.com/umair-nasir14/Word2World/assets/68095790/7b843e04-d009-4708-9b3e-686ddfe9c358)

#### Worlds:

![Untitled design](https://github.com/umair-nasir14/Word2World/assets/68095790/de351d5b-a8bf-4f11-8af9-a8eee1e45c33)

![world_1](https://github.com/umair-nasir14/Word2World/assets/68095790/5b85bb03-eed4-4879-ab07-4683d317ab20)
![world_2](https://github.com/umair-nasir14/Word2World/assets/68095790/6ccaa7e3-6573-4f20-b3a9-03e8992ffc9c)
![world_3](https://github.com/umair-nasir14/Word2World/assets/68095790/53e38643-d10a-4c16-a584-c0aa19116e60)
![world_4](https://github.com/umair-nasir14/Word2World/assets/68095790/fc8df4a5-63db-414f-96ca-a4094397ff9d)
![world_6](https://github.com/umair-nasir14/Word2World/assets/68095790/d92fa869-82de-4e97-bb77-2eb5fb7d04e2)
![world_7](https://github.com/umair-nasir14/Word2World/assets/68095790/751a753e-9e3d-41da-b146-fa852d0e7f1c)

### Note:

- The most stable model is `"gpt-4-turbo-2024-04-09"`.
- Currently only `OpenAI` models are supported.
- OS supported: `Windows`

### To-dos:

- [ ] Add support for Anthropic
- [ ] Add support for Groq
- [ ] Clean Code for easy integrations of new platforms, e.g. huggingface

### Cite:
```
@article{nasir2024word2world,
  title={Word2World: Generating Stories and Worlds through Large Language Models},
  author={Nasir, Muhammad U and James, Steven and Togelius, Julian},
  journal={arXiv preprint arXiv:2405.06686},
  year={2024}
}
```

