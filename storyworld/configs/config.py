from dataclasses import dataclass

@dataclass
class Config:
    """
    A configuration class for StoryWorrld.
    """


    experiment_number = "0001" 
    rounds = 3 # number of rounds to loop over
    model = "gpt-4-turbo-2024-04-09"

    story_paragraphs = [4, 5]
    total_objectives = 8

    save_dir = f"outputs/OpenEnded/exp_{experiment_number}"
    tile_data_dir = "storyworld/data"




