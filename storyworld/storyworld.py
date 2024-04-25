import os
import json
from dotenv import load_dotenv
from config.config import Config


cfg = Config()
load_dotenv()


if "gpt" in cfg.model: 
    import openai

else:
    raise NotImplementedError("Model not implemented yet!")





class StoryWorld:
    def __init__(self) -> None:
        self.total_input_tokens = []
        self.total_output_tokens = []
        self.worlds_history = {}
        self.previous_story = []
        self.previous_tile_map = []
        self.previous_map = []
        self.previous_eval = []
        self.prev_agent_reward = []
        self.total_spent = 0

    def run(self):
        
        

        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        if "gpt" in cfg.model: 
            from generators import OpenAIGenerator
            generator = OpenAIGenerator(self.total_input_tokens, self.total_output_tokens)
        
        
        story, story_prompt = generator.create_story(cfg.story_paragraphs, cfg.total_objectives)

        character_discriptions, character_discriptions_dict, character_prompt, protagonist_name, antagonist_name = generator.extract_character_info(story, story_prompt)
        tileset_discriptions, tileset_prompt = generator.extract_tileset_info(story, story_prompt, character_discriptions, character_prompt)
        tile_map_dict, tileset_map_discriptions, tileset_map_prompt = generator.map_tiles_to_chars(story, story_prompt, character_discriptions, 
                                                                                        character_prompt, tileset_discriptions, tileset_prompt)
        goal_discriptions, goal_prompt = generator.extract_goals(story, story_prompt, character_discriptions, character_prompt, 
                                                    tileset_discriptions, tileset_prompt, tileset_map_discriptions, tileset_map_prompt)
        important_tiles_list, important_tile_discriptions, important_tile_prompt = generator.extract_important_tiles(story, story_prompt, character_discriptions, character_prompt, 
                                                                                                        tileset_discriptions, tileset_prompt, tile_map_dict,
                                                                                                        tileset_map_discriptions, tileset_map_prompt, goal_discriptions,
                                                                                                            goal_prompt)
        walkable_tiles_list, walkable_tile_discriptions, walkable_tile_prompt = generator.extract_walkable_tiles(story, story_prompt, character_discriptions, character_prompt, 
                                                                                                    tileset_discriptions, tileset_prompt, tile_map_dict, 
                                                                                                    tileset_map_discriptions, tileset_map_prompt, goal_discriptions,
                                                                                                        goal_prompt)
        object_tiles_list, object_tile_discriptions, object_tile_prompt = generator.extract_interactive_object_tiles(story, story_prompt, character_discriptions, character_prompt, 
                                                                                                    tileset_discriptions, tileset_prompt, tile_map_dict, 
                                                                                                    tileset_map_discriptions, tileset_map_prompt, goal_discriptions,
                                                                                                        goal_prompt)
        #important_tiles_dict = find_elements_in_dict(tile_map_dict, important_tiles_list)
        #char_color_map, color_tiles_img = create_color_tile_map(tile_map_dict)
        for round in range(cfg.rounds):
            print(f"ROUND # {round}\n")
            world_map_fixed, world_map_fixed_with_chars, world_eval_dict, tileset_used_orig, tileset_used_dict, \
            char_color_map, color_tiles_img_with_char, story_paragraphs, total_objectives, good_feedback_check, bad_feedback_check, no_of_important_tiles, agent_reward, astar_path= generator.world_generation(round,
                                                                                                                                                self.previous_story,
                                                                                                                                                cfg.story_paragraphs,
                                                                                                                                                cfg.total_objectives,
                                                                                                                                                self.previous_tile_map,
                                                                                                                                                self.previous_map,
                                                                                                                                                self.previous_eval,
                                                                                                                                                story, 
                                                                                                                                                story_prompt, 
                                                                                                                                                character_discriptions,
                                                                                                                                                character_discriptions_dict, 
                                                                                                                                                character_prompt, 
                                                                                                                                                tileset_discriptions, 
                                                                                                                                                tileset_prompt,
                                                                                                                                                tile_map_dict, 
                                                                                                                                                tileset_map_discriptions, 
                                                                                                                                                tileset_map_prompt, 
                                                                                                                                                goal_discriptions, 
                                                                                                                                                goal_prompt, 
                                                                                                                                                important_tiles_list, 
                                                                                                                                                important_tile_discriptions, 
                                                                                                                                                important_tile_prompt, 
                                                                                                                                                walkable_tiles_list, 
                                                                                                                                                walkable_tile_discriptions, 
                                                                                                                                                walkable_tile_prompt,
                                                                                                                                                cfg.save_dir)


            
            self.previous_story.append(story['choices'][0]['message']['content'])
            self.previous_tile_map.append(tileset_used_dict)
            self.previous_map.append(world_map_fixed)
            self.previous_eval.append(world_eval_dict)
            self.prev_agent_reward.append(agent_reward)

            self.worlds_history[f"round_{round}"] = {"story": story['choices'][0]['message']['content'],
                                                        "character_information": character_discriptions['choices'][0]['message']['content'],
                                                        "tile_mapping": tileset_used_dict,
                                                        "goals": goal_discriptions['choices'][0]['message']['content'],
                                                        "important_tiles": important_tile_discriptions['choices'][0]['message']['content'],
                                                        "walkable_tiles": walkable_tile_discriptions['choices'][0]['message']['content'],
                                                        "interactive_object_tiles": object_tile_discriptions['choices'][0]['message']['content'],
                                                        "world_1st_layer": {"world":world_map_fixed, "tiles": tileset_used_orig},
                                                        "world": world_map_fixed_with_chars,
                                                        "evaluations": world_eval_dict,
                                                        "complexity": {
                                                            "good_feedback_check": good_feedback_check,
                                                            "bad_feedback_check": bad_feedback_check, 
                                                            "no_of_important_tiles": no_of_important_tiles,
                                                            "story_paragraphs": story_paragraphs,
                                                            "total_objectives": total_objectives
                                                        }}
            
            with open(cfg.save_dir +f"/data_gen_{cfg.experiment_number}.json", 'w') as f:
                json.dump(self.worlds_history, f)

            spent_this_gen = (sum(self.total_input_tokens)/1000)*0.01 + (sum(self.total_output_tokens)/1000)*0.03 
            self.total_spent += spent_this_gen
            print(f"$ spent on this gen = {spent_this_gen}")
            print(f"Total spent = {self.total_spent}")

if __name__ == "__main__":
    sw = StoryWorld()
    sw.run()
       





