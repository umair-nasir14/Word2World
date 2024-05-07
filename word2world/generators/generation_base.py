from word2world.utils import euclidean_distance

from word2world.solvers import find_characters, find_important_tiles

class Evaluator:
    def __init__(self, model, total_input_tokens, total_output_tokens):
        self.model = model
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens

    def tile_accuracy(self, map, world_eval_dictionary, important_tiles):

        characters = find_characters(map)
                
        total_characters = sum(1 for char in important_tiles if not char.isalpha())
            
        if len(characters) > 0:
            world_eval_dictionary["Accuracy_of_characters_placed"] = (len(characters)/total_characters) * 100
        else:
            world_eval_dictionary["Accuracy_of_characters_placed"] = 0
        
        important_tiles_found = find_important_tiles(map, important_tiles)
    
        if len(important_tiles_found) > 0:
            world_eval_dictionary["Accuracy_of_important_tiles_placed"] = (len(important_tiles_found)/len(important_tiles)) * 100
        else:
            world_eval_dictionary["Accuracy_of_important_tiles_placed"] = 0

        return world_eval_dictionary

    def euclidean_distance(self, map, previous_maps, world_eval_dictionary):

        euclidean_distance_score = 0
        if len(previous_maps) > 0:
            for previous_map in previous_maps:
                euclidean_distance_score += euclidean_distance(map, previous_map)
            euclidean_distance_score /= len(previous_maps)
        else:
            euclidean_distance_score = 0
        world_eval_dictionary["Average_euclidean_distance"] = euclidean_distance_score

        return world_eval_dictionary

    def evaluate_world(self):
        raise NotImplementedError("This method should be overridden by subclasses")

class Generator:
    def __init__(self, model, total_input_tokens, total_output_tokens):
        self.model = model
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens

    def create_story(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_character_info(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_tileset_info(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def map_tiles_to_chars(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_goals(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_important_tiles(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_walkable_tiles(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def extract_interactive_object_tiles(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def world_generation(self):
        raise NotImplementedError("This method should be overridden by subclasses")
   
    def action_generation(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    


