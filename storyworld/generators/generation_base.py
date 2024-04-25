class Evaluator:
    def __init__(self, model, total_input_tokens, total_output_tokens):
        self.model = model
        self.total_input_tokens = total_input_tokens
        self.total_output_tokens = total_output_tokens

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

    


