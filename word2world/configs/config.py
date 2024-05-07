from dataclasses import dataclass

@dataclass
class Config:
    """
    A configuration class for StoryWorrld.
    """

    model = "gpt-4-turbo-2024-04-09"
    story_paragraphs = [4, 5]
    total_objectives = 8
    rounds = 3 # number of rounds to loop over

    experiment_name = "Your_word2world" 
    save_dir = f"outputs/{experiment_name}"
    tile_data_dir = "word2world/data"




