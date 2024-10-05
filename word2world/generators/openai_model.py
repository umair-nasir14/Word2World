from word2world import Config
from word2world import Word2WorldEnv, LLMAgent
from word2world.utils import (
    extract_between_ticks,
    create_colored_tilemap_image,
    create_legend_image,
    create_legend,
    extract_present_elements,
    euclidean_distance,
    extract_list,
    extract_dict,
    get_image_color_tile_mapping,
    list_of_lists_to_string,
    find_character_position,
    overlap_dict,
    find_most_similar_images
    )
from word2world.fixers import remove_extra_special_chars, pad_rows_to_max_length
from word2world.solvers import find_characters, parse_grid, find_important_tiles, EnhancedAStarWorldAgent, WorldState
from .generation_base import Evaluator, Generator


import matplotlib.pyplot as plt
import numpy as np
import random
import json
import traceback
import time
import imageio
import pandas as pd
from PIL import Image
import openai
import traceback
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

cfg = Config()

class OpenAIEvaluator(Evaluator):
    def __init__(self, total_input_tokens, total_output_tokens):
        super().__init__(cfg.model, total_input_tokens, total_output_tokens)
    
    
    def evaluate_world(self, map, tile_map_dictionary, story, model, walkable_tiles, important_tiles, previous_maps):
        print(f"Evaluating World...")    
        no_of_exceptios = 0
        eval_system_prompt = "You are an evaluator of a 2D tilemap world created from a story. You extract meaning from a given story. You are also provided by Python dictionary-like mapping of tiles and characters or alphabets. You evaluate based on how the tiles have been placed and if they match how the story has explained. Your result being 'No' is not a bad thing, but it actually gives a good insight."
        evaluation_prompt = f"Given the story, tiles used to create the 2D map, and the 2D map, suggest whether the 2D tilemap world is coherent to the story. The story is as follows:\n{story['choices'][0]['message']['content']}\nThe tile mapping:{tile_map_dictionary}\nThe 2D tilemap world:\n{map}\n Check for all the tiles mentioned in the tile mapping being in the 2D tilemap world. Strictly return only 'Yes' or 'No' in your answer."
        done = False
        while not done:
            try:
                print(f"check#1 done = {done}")
                
                world_eval = openai.ChatCompletion.create(model=model, messages=[
                                                                        {"role": "system", "content": eval_system_prompt},
                                                                        {"role": "user", "content": evaluation_prompt}
                                                                        ], 
                                                                        temperature = 0.6)
                

                self.total_input_tokens.append(world_eval["usage"]["prompt_tokens"])
                self.total_output_tokens.append(world_eval["usage"]["completion_tokens"])
                
                world_eval_dictionary = extract_dict(world_eval['choices'][0]['message']['content'])    
                world_eval_dictionary = self.tile_accuracy(map, world_eval_dictionary, important_tiles)
                world_eval_dictionary = self.euclidean_distance(map, previous_maps, world_eval_dictionary)

                print("Evaluation: \n", world_eval_dictionary, "\n")
    
                done = True


            except Exception as e:
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n{tb}")
                no_of_exceptios += 1
                if no_of_exceptios >= 5:
                    done = True
                pass
        
        return world_eval_dictionary, self.total_input_tokens, self.total_output_tokens
    

class OpenAIGenerator(Generator):
    def __init__(self, total_input_tokens, total_output_tokens):
        self.model = cfg.model
        super().__init__(self.model, total_input_tokens, total_output_tokens)
    
    def create_story(self, story_paragraphs, total_objectives):
        print("Creating story...")
        print(f"Number of story paragraphs:{story_paragraphs}, Objectives of story: {total_objectives}")
        story_prompt = f"Write a {story_paragraphs[0]}-{story_paragraphs[1]} paragraph story which has characters including the protagonist trying to achieve something and the antagonist wanting to stop the protagonist. The story should describe an environment(s) where the story is set up. There should be {total_objectives} objectives for the protagonist in the story. One of them should be to defeat the antagonist somehow."
        story = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": story_prompt}])
        self.total_input_tokens.append(story["usage"]["prompt_tokens"])
        self.total_output_tokens.append(story["usage"]["completion_tokens"])
        print("\n")
        print(story['choices'][0]['message']['content'])
        print("\n")

        return story, story_prompt

    def extract_character_info(self, story, story_prompt):
        print("Extracting character information...")
        
        character_prompt = "Let's use the above story to create a 2D game. Write a specific description of each character which can be used as a prompt to generate sprites for the characters."
        character_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                    {"role": "user", "content": story_prompt},
                                                                                    {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": character_prompt},
                                                                                    ], 
                                                                                    temperature = 1)
        
        self.total_input_tokens.append(character_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(character_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(character_discriptions['choices'][0]['message']['content'])
        print("\n")
        
        character_dict_prompt = """Let's use the above story to create a 2D game. Create a Python dictionary that has keys as 'Protagonist', 'Antagonist' and any other character and values as very precise description. For example, the dictionary should look like:
        {
            'Protagonist': 'red dressed girl'
            'Antagonist': 'blue dressed boy'
        }
        The description should only be like this. Do not return in a Python response,
        """
        character_dict_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                    {"role": "user", "content": story_prompt},
                                                                                    {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": character_dict_prompt},
                                                                                    ], 
                                                                                    temperature = 1)
        self.total_input_tokens.append(character_dict_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(character_dict_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(character_dict_discriptions['choices'][0]['message']['content'])
        print("\n")

        antagonist_name = ""
        protagonist_name = ""

        character_discriptions_dict = extract_dict(character_dict_discriptions['choices'][0]['message']['content'])
        return character_discriptions, character_discriptions_dict, character_prompt, protagonist_name, antagonist_name

    def extract_tileset_info(self, story, story_prompt, character_discriptions, character_prompt):
        print("Extracting tileset information...")
        tileset_prompt = "Create an exhaustive list of tiles needed to create the environment."
        tileset_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                    {"role": "user", "content": story_prompt},
                                                                                    {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": character_prompt},
                                                                                    {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": tileset_prompt}
                                                                                    ], 
                                                                                    temperature = 1)
        self.total_input_tokens.append(tileset_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(tileset_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(tileset_discriptions['choices'][0]['message']['content'])
        print("\n")

        return tileset_discriptions, tileset_prompt

    def map_tiles_to_chars(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt):
        print("Mapping tiles to characters...")
        tileset_map_prompt = "Imagine each tile maps to an alphabet or a character. For environment use alphabets and for characters use special characters. Create it in a single Python Dictionary style. Return only and only a Python Dictionary and nothing else in your response. Don't return it in a Python response. Names should be the Keys and alphabets or characters should be the Values. Protagonist should always strictly be '@' and the antagonist should always strictly be '#'."
        tileset_map_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                    {"role": "user", "content": story_prompt},
                                                                                    {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": character_prompt},
                                                                                    {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": tileset_prompt},
                                                                                    {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                    {"role": "user", "content": tileset_map_prompt}
                                                                                    ], 
                                                                                    temperature = 1)
        self.total_input_tokens.append(tileset_map_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(tileset_map_discriptions["usage"]["completion_tokens"])
        print("\n")
        tile_map_dict = extract_dict(tileset_map_discriptions['choices'][0]['message']['content'])
        print(tileset_map_discriptions['choices'][0]['message']['content'])
        print("\n")

        return tile_map_dict, tileset_map_discriptions, tileset_map_prompt

    def extract_goals(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tileset_map_discriptions, tileset_map_prompt):
        print("Extracting goals..")
        goal_prompt = "What is the main goal for protagonist of the story? What are the small goals for protagonist to achieve the main goal of the story? Also create rewards and penalties based on the goals for protagonist. Create a score for each reward or penalty"
        goal_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                {"role": "user", "content": story_prompt},
                                                                                {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": character_prompt},
                                                                                {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_prompt},
                                                                                {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_map_prompt},
                                                                                {"role": "assistant", "content": tileset_map_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": goal_prompt}
                                                                                ], 
                                                                                temperature = 1)
        
        self.total_input_tokens.append(goal_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(goal_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(goal_discriptions['choices'][0]['message']['content'])
        print("\n")

        return goal_discriptions, goal_prompt

    def extract_important_tiles(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt):
        print("Extracting important tiles..")
        important_tile_prompt = f"Considering the above goals that you extracted from the story and the following tileset\n{tile_map_dict}\n create a Python list of the 15 most important characters of the tiles that should be placed in the 2D tilemap world. Remember, Protagonist, antagonist and non-player characters, if there are any, in the story will always be an important tiles. Only return the important tiles."
        important_tile_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                {"role": "user", "content": story_prompt},
                                                                                {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": character_prompt},
                                                                                {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_prompt},
                                                                                {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_map_prompt},
                                                                                {"role": "assistant", "content": tileset_map_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": goal_prompt},
                                                                                {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": important_tile_prompt}
                                                                                ], 
                                                                                temperature = 1)
        
        self.total_input_tokens.append(important_tile_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(important_tile_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(important_tile_discriptions['choices'][0]['message']['content'])
        important_tiles_list = extract_list(important_tile_discriptions['choices'][0]['message']['content'])
        print(important_tiles_list)
        print("\n")

        return important_tiles_list, important_tile_discriptions, important_tile_prompt

    def extract_walkable_tiles(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt):
        print("Extracting walkable tiles..")
        walkable_tile_prompt = f"Considering the above goals that you extracted from the story and the following tileset\n{tile_map_dict}\n create a Python list of the walkable tiles in the 2D tilemap world. Only return the walkable tiles."
        walkable_tile_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                {"role": "user", "content": story_prompt},
                                                                                {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": character_prompt},
                                                                                {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_prompt},
                                                                                {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_map_prompt},
                                                                                {"role": "assistant", "content": tileset_map_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": goal_prompt},
                                                                                {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": walkable_tile_prompt}
                                                                                ], 
                                                                                temperature = 1)
        
        self.total_input_tokens.append(walkable_tile_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(walkable_tile_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(walkable_tile_discriptions['choices'][0]['message']['content'])
        walkable_tiles_list = extract_list(walkable_tile_discriptions['choices'][0]['message']['content'])
        print(walkable_tiles_list)
        print("\n")

        return walkable_tiles_list, walkable_tile_discriptions, walkable_tile_prompt

    def extract_interactive_object_tiles(self, story, story_prompt, character_discriptions, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt):
        print("Extracting walkable tiles..")
        object_tile_prompt = f"Conseidering the above goals that you extracted from the story and the following tileset\n{tile_map_dict}\n create a Python list of the characters of the object tiles that can be interacted with in the 2D tilemap world. Only return the object tiles that can be interacted with."
        object_tile_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                {"role": "user", "content": story_prompt},
                                                                                {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": character_prompt},
                                                                                {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_prompt},
                                                                                {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": tileset_map_prompt},
                                                                                {"role": "assistant", "content": tileset_map_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": goal_prompt},
                                                                                {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
                                                                                {"role": "user", "content": object_tile_prompt}
                                                                                ], 
                                                                                temperature = 1)
        
        self.total_input_tokens.append(object_tile_discriptions["usage"]["prompt_tokens"])
        self.total_output_tokens.append(object_tile_discriptions["usage"]["completion_tokens"])
        print("\n")
        print(object_tile_discriptions['choices'][0]['message']['content'])
        object_tiles_list = extract_list(object_tile_discriptions['choices'][0]['message']['content'])
        print(object_tiles_list)
        print("\n")

        return object_tiles_list, object_tile_discriptions, object_tile_discriptions

    def world_generation(self, rounds, previous_story, story_paragraphs, total_objectives, previous_tile_map, previous_map, previous_eval, story, story_prompt, character_discriptions, character_discriptions_dict, character_prompt, tileset_discriptions, tileset_prompt, tile_map_dict, tileset_map_discriptions, tileset_map_prompt, goal_discriptions, goal_prompt, important_tiles_list, important_tile_discriptions, important_tile_prompt, walkable_tiles_list, walkable_tile_discriptions, walkable_tile_prompt, save_dir):
        print(f"Generating World...")
        NO_OF_EXCEPTIONS_2 = 0
        no_of_important_tiles = 15
        history_to_keep = 0
        good_feedback_prompt = ""
        bad_feedback_prompt = ""
        good_feedback_check = 0
        bad_feedback_check = 0
        if rounds == 0 or history_to_keep > 0:
            world_system_prompt = "You are a 2D game designer that is profficient in designing tile-based maps. Designing any size of the tile-based map is not a problem for you. This is your first round of generation. You are given the goals to achieve and a list of important tiles to place. Consider them to make the world. Do not place the protagonist, the antagonist and the interactive objects of the story right now. Only create the world right now. Also, consider goals that you extracted earlier and generate while keeping them in context."    
            world_prompt = f"Using the following tile to character mapping:\n{tile_map_dict}\nCreate an entire world on a tile-based grid. Do not create things that would neew more than one tile. For example, a house or a building needs more than one tile to be made. Also, following characters are important to place:\n{important_tiles_list}\n and walkable tiles:\n{walkable_tiles_list}\n. Use {no_of_important_tiles} important tiles to create the world. Do not place the protagonist, the antagonist and the interactive objects of the story right now. Only create the world right now. Create it is a string format with three backticks to start and end with (```) and not in a list format."
        else:
            if len(previous_map) > history_to_keep:
                previous_map = previous_map[-history_to_keep:]
                previous_story = previous_story[-history_to_keep:]
                previous_tile_map = previous_tile_map[-history_to_keep:]
                previous_eval = previous_eval[-history_to_keep:]
            history_intro = f"For your reference, here are the previous {len(previous_map)} stories, their tile mapping and corresponding 2D world maps\n"
            for i in range(len(previous_map)):
                history = history_intro + f"Story {i}: {previous_story[i]}\nTile Map for story {i}:\n{previous_tile_map[i]}\n, 2D World map for story {i}:\n {previous_map[i]} and evaluation for the 2D World map:\n{previous_eval[i]}. {good_feedback_prompt}\n{bad_feedback_prompt} Create higher quality and with a higher diversity map."
            world_system_prompt = f"You are a 2D game designer that is profficient in designing tile-based maps. Designing any size of the tile-based map is not a problem for you. This is the generation number {round} for you and you will be provided by previous generation results. Improve evaluation scores in each generation. Previous evaluation scores will be provided to you. You are given the goals to achieve and a list of important tiles to place. Additionally you are given 2D tile-maps and stories that were create before for you to make a better map. Consider them to make the world. Do not place the protagonist, the antagonist and the important objects of the story right now. Only create the world right now. Also, consider goals that you extracted earlier and generate while keeping them in context."    
            world_prompt = f"Using the following tile to character mapping:\n{tile_map_dict}\nCreate an entire world on a tile-based grid. Do not create things that would neew more than one tile. For example, a house or a building needs more than one tile to be made. Also, following characters are important to place:\n{important_tiles_list}\n and walkable tiles:\n{walkable_tiles_list}\n Use {no_of_important_tiles} important tiles to create the world. Do not place the protagonist, the antagonist and the important objects of the story right now. Only create the world right now. Create it is a string format with three backticks to start and end with (```) and not in a list format. {history}"
        done = False
        while not done:
            try:
                world_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                        {"role": "user", "content": story_prompt},
                                                                                        {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": character_prompt},
                                                                                        {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": tileset_prompt},
                                                                                        {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": tileset_map_prompt},
                                                                                        {"role": "assistant", "content": tileset_map_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": goal_prompt},
                                                                                        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": important_tile_prompt},
                                                                                        {"role": "assistant", "content": important_tile_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "system", "content": world_system_prompt},
                                                                                        {"role": "user", "content": world_prompt}
                                                                                        ], 
                                                                                        temperature = 1)

                print("World: \n")
                print(world_discriptions['choices'][0]['message']['content'])
                print("\n")
                self.total_input_tokens.append(world_discriptions["usage"]["prompt_tokens"])
                self.total_output_tokens.append(world_discriptions["usage"]["completion_tokens"])
                print("Extracting tilemap..")
                print("\n")

                world_map_raw = extract_between_ticks(world_discriptions['choices'][0]['message']['content'])
                print(world_map_raw)
                print("\n")
                print("Fixing tilemap..")
                world_map_raw = world_map_raw.replace(' ', '').replace('"', '')
                world_map_fixed = remove_extra_special_chars(world_map_raw)
               
                print(world_map_fixed)
                used_char_dict = extract_present_elements(world_map_fixed, tile_map_dict)
                
                world_with_characters_prompt = f"Now that you have created the following world map:\n{world_map_fixed}\n Place only the protagonist, the antagonist and the interactive objects of the story. Do not change anything in the world, just place only the protagonist, the antagonist and the interactive objects in the world."
                world_with_characters_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                        {"role": "user", "content": story_prompt},
                                                                                        {"role": "assistant", "content": story['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": character_prompt},
                                                                                        {"role": "assistant", "content": character_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": tileset_prompt},
                                                                                        {"role": "assistant", "content": tileset_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": tileset_map_prompt},
                                                                                        {"role": "assistant", "content": tileset_map_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": goal_prompt},
                                                                                        {"role": "assistant", "content": goal_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": important_tile_prompt},
                                                                                        {"role": "assistant", "content": important_tile_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "system", "content": world_system_prompt},
                                                                                        {"role": "user", "content": world_prompt},
                                                                                        {"role": "assistant", "content": world_discriptions['choices'][0]['message']['content']},
                                                                                        {"role": "user", "content": world_with_characters_prompt},
                                                                                        ], 
                                                                                        temperature = 1)

                print("World: \n")
                print(world_with_characters_discriptions['choices'][0]['message']['content'])
                print("\n")
                self.total_input_tokens.append(world_with_characters_discriptions["usage"]["prompt_tokens"])
                self.total_output_tokens.append(world_with_characters_discriptions["usage"]["completion_tokens"])
                print("Extracting tilemap..")
                print("\n")

                world_map_raw_with_chars = extract_between_ticks(world_with_characters_discriptions['choices'][0]['message']['content'])
                print(world_map_raw_with_chars)
                print("\n")
                print("Fixing tilemap..")
                world_map_raw_with_chars = world_map_raw_with_chars.replace(' ', '').replace('"', '')
                world_map_fixed_with_chars = remove_extra_special_chars(world_map_raw_with_chars)
                print(world_map_fixed_with_chars)


                evaluator = OpenAIEvaluator(self.total_input_tokens, self.total_output_tokens)
                world_eval_dict, self.total_input_tokens, self.total_output_tokens = evaluator.evaluate_world(map=world_map_fixed_with_chars,
                                                                                                              tile_map_dictionary=tile_map_dict,
                                                                                                              story=story,
                                                                                                              model=self.model,
                                                                                                              walkable_tiles=walkable_tiles_list,
                                                                                                              important_tiles=important_tiles_list,
                                                                                                              previous_maps=previous_map)
                

                llm_agent_reward, astar_path, objectives = self.action_generation(rounds,story['choices'][0]['message']['content'],"protagonist","antagonist", character_discriptions_dict,world_map_fixed,world_map_fixed_with_chars,used_char_dict,tile_map_dict,"color_tiles_img_with_char",
                        "char_color_map",walkable_tile_discriptions['choices'][0]['message']['content'],important_tile_discriptions['choices'][0]['message']['content'],goal_discriptions['choices'][0]['message']['content'], save_dir)
                
                
                world_eval_dict["agent_reward"] = llm_agent_reward
                world_eval_dict["astar_path"] = astar_path


                story_paragraphs, total_objectives, no_of_important_tiles, bad_feedback_prompt, good_feedback_prompt = self.feedback_checks(rounds, world_eval_dict, previous_eval, story_paragraphs, total_objectives, no_of_important_tiles)
                    
                done = True

            except Exception as e:
                
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n {tb}")
                NO_OF_EXCEPTIONS_2 += 1

                if NO_OF_EXCEPTIONS_2 >= 2:
                    done = True
                pass

        color_tiles_img_with_char = ""

        return world_map_fixed, world_map_fixed_with_chars, world_eval_dict, used_char_dict, tile_map_dict, color_tiles_img_with_char, \
                color_tiles_img_with_char, story_paragraphs, objectives, total_objectives, good_feedback_check, bad_feedback_check, no_of_important_tiles, llm_agent_reward, astar_path


    def action_generation(self, round,
                        story,
                        protagonist_name,
                        antagonist_name,
                        character_discriptions_dict,
                        world_map_fixed,
                        world_map_fixed_with_chars,
                        tileset_used_dict_1st_layer,
                        tileset_used_dict,
                        color_tiles_img_with_char,
                        char_color_map,
                        walkable_tiles_list,
                        object_tiles_list,
                        goal_discriptions, 
                        save_dir):

        
        print("Generating Actions...")
        except_done = False
        NO_OF_EXCEPTIONS_3 = 0
        total_reward = 0
        frames = [] 
        total_episodes = 1
        episodes = 0
        all_episodes_rewards = []
        try:
            while not except_done:
                
                
                objective_tile_system = "You are a great planner in 2D game. You plan objectives for the protagonist of the game. All objectives should match the goals extracted from the story. Objectives should strictly follow them. Return a Python dictionary of the objective as the key and a tile that achieves the objective and the position of the tile. For example 'Objective': ['A', 6, 1]. Only return a Python dictionary. Do not return a python response."
                objective_tile_prompt = f"Given the story:\n{story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping:\n{tileset_used_dict}\n You are also provided with the description of the goals:\n{goal_discriptions}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n Taking this information into your context, create the objectives to achieve and also provide the tile that you will pick up, reach the position at or hit the enemy. Return a Python dictionary of the objective as the key and a tile that achieves the objective and the position of the tile. The pattearn should be 'Objective': ['tile', row, column], for example 'Objective': ['A', 6, 1], thus the first element would be the tile and second and third elements of the list will be position of the tile. Return strictly in this format."
                
                objective_tile_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                        {"role": "system", "content": objective_tile_system},
                                                                                        {"role": "user", "content": objective_tile_prompt}
                                                                                        ])
                
                
                

                print("Objectives: \n")
                print(objective_tile_discriptions['choices'][0]['message']['content'])
                print("\n")
                
                objective_tile_dict = extract_dict(objective_tile_discriptions['choices'][0]['message']['content'])

                print("objective_tile in a dict: \n")
                print(objective_tile_dict)
                print("\n")

                self.total_input_tokens.append(objective_tile_discriptions["usage"]["prompt_tokens"])
                self.total_output_tokens.append(objective_tile_discriptions["usage"]["completion_tokens"])
                total_actions = {}
                objective_flag = False

                walkable_tiles_list = extract_list(walkable_tiles_list)
                object_tiles_list = extract_list(object_tiles_list)

                # ASTAR Search
                world_map_fixed_with_chars = pad_rows_to_max_length(world_map_fixed_with_chars)
                parsed_world_map = parse_grid(world_map_fixed_with_chars)

                objective_tile_list = []
                for _keys, str_obj in objective_tile_dict.items():
                    temp_list = extract_list(str(str_obj))
                    objective_tile_list.append((temp_list[1], temp_list[2]))
                
                solving = False
                solving_exceptions = 0
                while not solving:
                    try:
                        # Initialize the game state and agent
                        game_state = WorldState(walkable_tiles_list, object_tiles_list, parsed_world_map, objective_tile_list)
                        game_state = game_state.stringInitialize(parsed_world_map, objective_tile_list)
                        astar_agent = EnhancedAStarWorldAgent(walkable_tiles_list, objective_tile_list, game_state, object_tiles_list)
                        astar_path, _, _, game_map_updated, _ = astar_agent.getSolution(game_state,maxIterations=10000)
                        print(f"astar_path: {len(astar_path)}")
                        solving = True
                    except Exception as e:
                        #print(f"check#3 done = {done}")
                        tb = traceback.format_exc()
                        print(f"Exception raised: {e}\n {tb}")
                        solving_exceptions += 1
                        if solving_exceptions >= 5:
                            solving = True
                            astar_path = []
                            print(f"astar_path: {len(astar_path)}")
                        pass
                    

                removed_value = tileset_used_dict.pop('Protagonist', None) 
                removed_value = tileset_used_dict.pop('Antagonist', None) 
                tileset_used_dict[character_discriptions_dict["Protagonist"]] = "@"
                tileset_used_dict[character_discriptions_dict["Antagonist"]] = "#"
                print("Retrieving images.")
                tile_images,_s= find_most_similar_images(tileset_used_dict,cfg.tile_data_dir)
                print("Images Retrieved.")
                tile_images_1st_layer = overlap_dict(tile_images, tileset_used_dict_1st_layer)

                legend = create_legend(tile_images,tileset_used_dict)
                #plt.imshow(legend)
                #plt.axis('off')
                #plt.savefig(save_dir + f'/world_legend_with_chars_{round}.png', format='png', dpi=150, bbox_inches='tight')
                #plt.show()
                legend.save(save_dir + f'/world_legend_with_chars_{round}.png', 'PNG')

                env = Word2WorldEnv(walkable_tiles_list, tile_images_1st_layer, tile_images, world_map_fixed, world_map_fixed_with_chars, object_tiles_list, "#")
                agent = LLMAgent()
                state = env.reset()
                env_image = env.render(mode="image")
                env_image.save(save_dir + f'/world_map_with_chars_{round}.png', 'PNG')

                reward_feedback = "This is your first objective"
                reward_design = {
                    "You are 8 tiles away from objective thus objective is incomplete": -100,
                    "You are 5 to 8 tiles away from objective thus objective is incomplete": -50,
                    "You are 3 to 5 tiles away from objective": +25,
                    "You are 1 to 3 tiles away from objective": +50,
                    "You are 1 tile away or your are on the objective tile from objective": +100,
                    "You have completed the objective": +100,
                }
                protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                
                prev_episode_reward = 0
                done = False
                
                while not done:
                    reward = 0
                    for i in range(len(objective_tile_dict)):
                        
                        print("\n")
                        print(f"OBJECTIVE: {list(objective_tile_dict.keys())[i]}")
                        print("\n")
                        total_actions[list(objective_tile_dict.keys())[i]] = []
                        #while not objective_flag:
                        action_system = f"You are a great planner in 2D game. You plan actions for the protagonist of the game to achieve all objects. You are given objectives, tiles and the position of tiles to achieve the objectives. You have the following options as actions: 'move_up', move_down, 'move_right', 'move_left', 'pick_object', 'hit_enemy'. Generate a sequence of actions that will achieve the objective. Only return the sequence of actions from the options."
                        action_prompt = f"Given the story:\n{story}\n a 2D tile map of a world was created for the story:\n{world_map_fixed_with_chars}\n The tile map was created using the following tile to character mapping which has information about all the tiles:\n{tileset_used_dict}\n You are also provided with a set of objectives:\n{objective_tile_dict}\n and walkable tiles:\n{walkable_tiles_list}\n and interactive object tiles:\{object_tiles_list}\n. The character '@' is the protagonist of the story and you are controlling it. The current position of protagonist is {protagonist_position}. The rewards will be given as follows:\n{reward_design}\n{reward_feedback}. Accumulative rewards for all the previous objectives tille now are {reward}. Taking this information into your context, create a sequence of actions for the protagonist to complete the objective: {list(objective_tile_dict.keys())[i]}, which is to reach the tile, 'pick_object' or 'hit_enemy' at tile and position: {list(objective_tile_dict.values())[i]}. Strictly return a Python dictionary with the entry as 'action'. Only return Python dictionary. Do not return it in a Python response."
                        
                        actions_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                                {"role": "system", "content": action_system},
                                                                                                {"role": "user", "content": action_prompt}
                                                                                                ])
                        
                        self.total_input_tokens.append(actions_discriptions["usage"]["prompt_tokens"])
                        self.total_output_tokens.append(actions_discriptions["usage"]["completion_tokens"])
                        action_dict = extract_dict(actions_discriptions['choices'][0]['message']['content'])
                        print("Action: \n")
                        print(action_dict["action"])
                        print("\n")
                        total_actions[list(objective_tile_dict.keys())[i]].append(action_dict["action"])
                        
                        
                        for action_str in action_dict["action"]:
                            action = agent.action(action_str)
                            state, _r, done, _ = env.step(action)
                            
                            frame = env.render(mode='rgb_array')  # Capture the frame
                            frames.append(frame)  # Append the frame
                            time.sleep(0.01)
                    
                    
                        current_state = list_of_lists_to_string(state)
                        
                        print(current_state)
                        print("\n")
                        
                        check_prompt = f"Given the previous world state:\n{world_map_fixed_with_chars}\n and the updated state that you returned: \n{current_state}\n is the objective {list(objective_tile_dict.keys())[i]} completed? Remember, from the dictionary of objectives, this objective will be completed when you reach tile {list(objective_tile_dict.values())[0]} at position {list(objective_tile_dict.values())[1]} or you are one tile aound this position in any directions. Strictly, only return 'Complete' or 'Incomplete'."
                        
                        check_discriptions = openai.ChatCompletion.create(model=self.model, messages=[
                                                                                                {"role": "system", "content": action_system},
                                                                                                {"role": "user", "content": action_prompt},
                                                                                                {"role": "assistant", "content": actions_discriptions['choices'][0]['message']['content']},
                                                                                                {"role": "user", "content": check_prompt}
                                                                                                ])
                        world_map_fixed_with_chars = current_state
                        
                        self.total_input_tokens.append(check_discriptions["usage"]["prompt_tokens"])
                        self.total_output_tokens.append(check_discriptions["usage"]["completion_tokens"])
                        
                        for k, value in enumerate(objective_tile_dict.values()):
                            if k == i:
                                objective_pos = extract_list(str(value))
                        protagonist_position = find_character_position(world_map_fixed_with_chars, "@")
                        print("\n")
                        print(f"protagonist_position: {protagonist_position}")
                        print(f"objective_position: [{objective_pos[1]},{objective_pos[2]}]")
                        

                        distance_from_objective = (abs(objective_pos[1] - protagonist_position[0]), abs(objective_pos[2] - protagonist_position[1]))
                        print(f"distance from current objective: [{distance_from_objective[0]}, {distance_from_objective[1]}]") 
                        print("\n")
                        reward_feedback = ""
                        reward_feedback = "Your previous objectives reward feedback is: "
                        if (distance_from_objective[0] > 8 or distance_from_objective[1] > 8):
                            reward -= 100
                            reward_feedback += f"You were very far from the objective tile so you were given a regret(negative reward) of -100 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] > 5 and distance_from_objective[0] < 8) or (distance_from_objective[1] > 5 and distance_from_objective[1] < 8):
                            reward -= 50
                            reward_feedback += f"You were far from the objective tile so you were given a regret(negative reward) of -50 points and objective was INCOMPLETE"
                        if (distance_from_objective[0] <= 5 and distance_from_objective[0] > 3) and (distance_from_objective[1] <= 5 and distance_from_objective[1] > 3):
                            reward += 25
                            reward_feedback += f"You were close to the objective tile so you were given a reward of 25 points"
                        if (distance_from_objective[0] < 3 and distance_from_objective[0] > 1) and (distance_from_objective[1] < 3 and distance_from_objective[1] > 1):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1) and (distance_from_objective[1] > 1 and distance_from_objective[1] <= 5):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"
                        if (distance_from_objective[1] <= 1) and (distance_from_objective[0] > 1 and distance_from_objective[0] <= 5):
                            reward += 50
                            reward_feedback += f"You were very close to the objective tile so you were given a reward of 50 points"

                        if (distance_from_objective[0] <= 1 and distance_from_objective[1] <= 1) or check_discriptions['choices'][0]['message']['content'] == "Complete":
                            
                            if (distance_from_objective[0] == 0 and distance_from_objective[1] == 0) and check_discriptions['choices'][0]['message']['content'] == "Complete":
                                reward += 200
                                #objective_flag = True
                                reward_feedback += f"You were by the objective tile and you COMPLETED the objective so you were given a reward of 200 points"
                            else:
                                reward += 100
                                reward_feedback += f"You were by the objective tile so you were given a reward of 100 points"
                        print("\n")
                        print(f"EPISODE REWARDS uptill now: {reward}")
                        print("\n")
                    total_reward += reward
                    episodes += 1
                    all_episodes_rewards.append(reward)
                    print("\n")
                    print(f"TOTAL REWARD for EPISODE: {total_reward}")
                    if episodes == total_episodes:
                        done = True
                
                    with imageio.get_writer(f'{cfg.save_dir}_{round}.mp4', fps=10) as video:
                        for frame in frames:
                            video.append_data(frame)

                except_done = True
            
        except Exception as e:
            #print(f"check#3 done = {done}")
            tb = traceback.format_exc()
            print(f"Exception raised: {e}\n {tb}")
            NO_OF_EXCEPTIONS_3 += 1
            if NO_OF_EXCEPTIONS_3 >= 5:
                except_done = True
            pass

        if len(all_episodes_rewards) == 0:
            all_episodes_rewards.append(0)
        
        return max(all_episodes_rewards), len(astar_path), objective_tile_dict
    
