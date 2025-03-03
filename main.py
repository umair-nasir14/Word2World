from word2world import Word2World
from dotenv import load_dotenv
import argparse

from word2world import Config


def main():
    config = Config()

    parser = argparse.ArgumentParser(description="Process Word2World inputs.")
    parser.add_argument('--model', type=str, help='Defaults to "gpt-4-turbo-2024-04-09". Currently supports gpt-4 and gpt-3.5.')
    parser.add_argument('--min_story_paragraphs', type=int, help='Defaults to "4". Provide an int, which is the minimum number of paragraphs')
    parser.add_argument('--max_story_paragraphs', type=int, help='Defaults to "5". Provide an int, which is the maximum number of paragraphs')
    parser.add_argument('--total_objectives', type=int, help='Dafualts to 8. Used to decide number of objectives in the story.')
    parser.add_argument('--rounds', type=str, help='Defaults to 3. Used to decide rounds of world generation.')
    parser.add_argument('--experiment_name', type=str, help='Defaults to "Your_word2world".')
    parser.add_argument('--save_dir', type=str, help='Defaults to "outputs/{--experiment_name}"')
    args = parser.parse_args()

    if args.model:
        config.model = args.model
    if args.min_story_paragraphs and args.max_story_paragraphs:
        if (args.min_story_paragraphs > args.max_story_paragraphs):
            raise ValueError("Minimum number of paragraphs should be less than maximum number of paragraphs.") 
        if (args.max_story_paragraphs < args.min_story_paragraphs):
            raise ValueError("Maximum number of paragraphs should be greater than minimum number of paragraphs.") 
        config.story_paragraphs = [int(args.min_story_paragraphs),int(args.max_story_paragraphs)]
    if args.total_objectives:
        config.total_objectives = int(args.total_objectives)
    if args.rounds:
        config.rounds = int(args.rounds)
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.save_dir:
        config.save_dir = args.save_dir
    
    world = Word2World()
    world.run(config)

if __name__ == "__main__":
    main()