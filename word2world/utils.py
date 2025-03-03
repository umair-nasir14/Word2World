
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import ast
import csv
import os
import re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import Dict, Tuple
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModel
import torch
from scipy.spatial.distance import cosine
import traceback

def load_image_dict(char_tile_mapping, folder_path):
    """
    Load images based on a character-to-file mapping.

    Args:
    - char_tile_mapping (dict): A dictionary mapping descriptive names to single characters.
    - folder_path (str): Path to the folder containing the images.

    Returns:
    - dict: A dictionary with characters as keys and loaded image objects as values.
    """
    image_dict = {}
    for description, char in char_tile_mapping.items():
        # Construct the file path assuming the images are named after the dictionary values
        # and have a common image extension (e.g., .png)
        file_path = os.path.join(folder_path, f"{char}.png")

        # Check if the file exists before attempting to load
        if os.path.exists(file_path):
            image = Image.open(file_path)
            mode = image.mode
            if 'A' not in mode:
                image = image.convert('RGBA')
            image_dict[char] = image
        else:
            print(f"Image file for {description} ({char}.png) not found in {folder_path}.")

    return image_dict


def grid_to_csv(grid, output_file_path):
    lines = [line for line in grid.strip().split('\n')]

    # Write to the CSV file
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for line in lines:
            writer.writerow(list(line))

    return output_file_path

def dict_to_txt_file(dictionary, file_path):
    """
    Write the key-value pairs of a dictionary to a text file.
    :param dictionary: dict, the dictionary to write to the file
    :param file_path: str, the path to the text file where the dictionary will be written
    """
    with open(file_path, 'w') as file:
        for key, value in dictionary.items():
            file.write(f'"{key}":"{value}"\n')


#def extract_between_ticks(text: str) -> str:
#    """
#    Extracts text between two sets of triple backticks (```) in the given text.
#    Raises an error if the triple backticks are not found.
#    """
#    # Split the text by triple backticks
#    parts = text.split("```")
#    
#    # If there are at least three parts (beginning, desired text, end), return the desired text
#    if len(parts) >= 3:
#        return parts[1]
#    else:
#        raise ValueError("Triple backticks (```) not found or text between them is missing.")
def extract_between_ticks(text):
    """
    Extracts text between the first two sets of triple backticks (```) in the given text.
    Raises an error if the triple backticks are not found or if the text between them is missing.
    """
    # Split the text by triple backticks
    parts = text.split("```")
    
    # If there are at least three parts (beginning, desired text, end), return the desired text
    if len(parts) >= 3 and parts[1].strip():
        return parts[1].strip()
    else:
        raise ValueError("Triple backticks (```) not found or text between them is missing.")


def merge_dictionaries(A, B):
    # Create a new dictionary to hold the merged data
    merged_dict = {}

    # Iterate through each item in dictionary A
    for key, value in A.items():
        # If the value from A is a key in B, add to merged_dict with the new key
        if value in B:
            merged_dict[key] = B[value]

    return merged_dict

def assign_random_color(assigned_colors):
    """ Generate a random color in hex format. """
    for i in range(10):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if color in assigned_colors:
            continue
        else:
            return color

def get_all_tile_chars(tile_dictionary):
    return ''.join(tile_dictionary.values())

def create_color_tile_map(tile_map_dict):
    tile_map = get_all_tile_chars(tile_map_dict)
    unique_chars = set("".join(tile_map))
    assigned_colors = []
    char_color_map = {}
    for char in unique_chars:
        color = assign_random_color(assigned_colors)
        char_color_map[char] = color
        assigned_colors.append(color)
    #char_color_map = {char: assign_random_color(assigned_colors) for char in unique_chars}

    # Create an image with colored tiles
    tile_size = 20 # Size of each tile in the image
    num_tiles = len(unique_chars)
    img = np.zeros((tile_size, tile_size * num_tiles, 3), dtype=np.uint8)

    for i, char in enumerate(unique_chars):
        # Convert hex color to RGB
        hex_color = char_color_map[char].lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        img[:, i * tile_size:(i + 1) * tile_size] = rgb_color

    return char_color_map, img

def get_image_color_tile_mapping(char_color_map):

    char_image_map = {}
    size = (64, 64)  # Define the size of the image

    for char, color in char_color_map.items():
        image = Image.new('RGBA', size, color)
        char_image_map[char] = image

    return char_image_map

def create_colored_tilemap_image(tile_map, char_color_map, tile_size=64):
    # Split the tile map into lines and find its dimensions
    lines = tile_map.strip().split('\n')
    height, width = len(lines), max(len(line) for line in lines)

    # Create an empty image with a larger tile size
    img = np.zeros((height * tile_size, width * tile_size, 3), dtype=np.uint8)

    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            # Convert hex color to RGB and fill the tile
            hex_color = char_color_map[char].lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            img[y * tile_size:(y + 1) * tile_size, x * tile_size:(x + 1) * tile_size] = rgb_color

    return img



def create_legend(dict_A, dict_B):
    # Initialize a blank image with enough space for the legend. 
    # Adjust the width and height accordingly based on your content and number of items.
    legend_width = 600
    legend_height = 100 * len(dict_B)  # 32 pixels per item
    legend = Image.new('RGB', (legend_width, legend_height), 'white')
    
    # Load a font.
    font = ImageFont.load_default(size=40)
    
    draw = ImageDraw.Draw(legend)
    
    # Iterate through dict_B to create each line of the legend
    y_offset = 0
    for description, character in dict_B.items():
        # Resize the image from dict_A to 32x32
        image = dict_A[character].resize((100, 100))
        
        # Paste the image onto the legend
        legend.paste(image, (0, y_offset))
        
        # Calculate the text position and draw the description
        text_position = (135, y_offset + 8)  # Adjust text positioning as needed
        draw.text(text_position, description, fill='black', font=font)
        
        # Move to the next item position
        y_offset += 100
    
    return legend

def create_legend_image(char_color_map, tile_map_dict, tile_size=40):
    """ Create an image that serves as a legend with text on the left and color tiles on the right. """
    # Number of unique characters
    num_chars = len(char_color_map)

    # Create a figure and axis for matplotlib
    fig, ax = plt.subplots(figsize=(4, num_chars / 2))  # Adjusted figure width for better visibility
    ax.set_xlim([0, tile_size * 4])
    ax.set_ylim([0, tile_size * num_chars])

    merged_char_color_dict = merge_dictionaries(tile_map_dict, char_color_map)

    for i, (char, color) in enumerate(merged_char_color_dict.items()):
        # Convert hex color to RGB
        hex_color = color.lstrip('#')
        rgb_color = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))

        # Draw a rectangle for the text background
        text_bg_rect = plt.Rectangle((0, i * tile_size), tile_size * 3, tile_size, color='black')
        ax.add_patch(text_bg_rect)

        # Add the character text in white color
        ax.text(10, (i + 0.5) * tile_size, char, fontsize=12, color='white')

        # Draw a rectangle for the color tile
        color_rect = plt.Rectangle((tile_size * 3, i * tile_size), tile_size, tile_size, color=color)
        ax.add_patch(color_rect)

    ax.axis('off')
    return fig

def convert_response_to_dict(input_string):
    # Preprocess the string to make it valid JSON
    # Escape single quotes within the values and replace outer single quotes with double quotes
    json_string = input_string.strip()
    json_string = json_string.replace("': '", '": "').replace("',\n    '", '",\n    "').replace("{\n    '", '{\n    "').replace("'\n}", '"\n}').replace("python","")

    try:
        # Parse the JSON string into a Python dictionary
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Return the error if the string cannot be parsed
        return str(e)

def find_elements_in_dict(dict_a, list_b):
    # Create an empty dictionary to store the matches
    found_elements = {}
    # Iterate through the dictionary items
    for key, value in dict_a.items():
        # If the value is in list_b, add the key-value pair to the found_elements dictionary
        if value in list_b:
            found_elements[key] = value
    return found_elements

def extract_present_elements(grid_string, elements_dict):
    # Initialize an empty dictionary to hold the elements present in the grid
    present_elements = {}
    
    # Iterate over the dictionary items
    for key, value in elements_dict.items():
        # Check if the element's symbol is present in the grid string
        if value in grid_string:
            # If present, add the key to the present_elements dictionary
            present_elements[key] = value
            
    return present_elements


def euclidean_distance(a, b):
    # Convert strings to lists of ASCII values
    ascii_a = np.array([ord(char) for char in a])
    ascii_b = np.array([ord(char) for char in b])
    
    # If lengths differ, truncate the longer one to match the shorter one
    min_len = min(len(ascii_a), len(ascii_b))
    ascii_a = ascii_a[:min_len]
    ascii_b = ascii_b[:min_len]
    
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((ascii_a - ascii_b) ** 2))
    
    return distance

def map_to_list(str_map):
    return str_map.splitlines()

def extract_list(string_data):
    try:
        # Search for the list pattern in the string, including those within ```json and ```python blocks
        pattern = r'```(?:json|python)?\s*(\[.*?\])\s*```|(\[.*?\])'
        matches = re.findall(pattern, string_data, re.DOTALL)
        if matches:
            # Flatten the matches and filter out empty strings
            matches = [match[0] or match[1] for match in matches if match[0] or match[1]]
            # Iterate over all matches to find the first valid list
            for match in matches:
                try:
                    # Try to parse as Python list
                    extracted_list = ast.literal_eval(match)
                    if isinstance(extracted_list, list):
                        return extracted_list
                except (ValueError, SyntaxError):
                    try:
                        # Try to parse as JSON list
                        extracted_list = json.loads(match)
                        if isinstance(extracted_list, list):
                            return extracted_list
                    except json.JSONDecodeError:
                        continue
            print("No valid list found in the matches.")
            return []
        else:
            print("No list-like pattern found in the string.")
            return []
    except ValueError as e:
        print(f"Error converting string to list: {e}")
        return []
    except SyntaxError as e:
        print(f"Syntax error in the string: {e}")
        return []

def list_of_lists_to_string(lists):
    return '\n'.join([''.join(sublist) for sublist in lists])

def extract_dict(string_data):
    try:
        # Search for the dictionary pattern in the string, including those within ```json and ```python blocks
        pattern = r'```(?:json|python)?\s*(\{.*?\})\s*```|(\{.*?\})'
        matches = re.findall(pattern, string_data, re.DOTALL)
        if matches:
            # Flatten the matches and filter out empty strings
            matches = [match[0] or match[1] for match in matches if match[0] or match[1]]
            # Iterate over all matches to find the first valid dictionary
            for match in matches:
                try:
                    # Try to parse as Python dictionary
                    mission_dict = ast.literal_eval(match)
                    if isinstance(mission_dict, dict):
                        # Add single quotes to string values
                        for key, value in mission_dict.items():
                            if isinstance(value, str):
                                mission_dict[key] = f"{value}"
                        return mission_dict
                except (ValueError, SyntaxError):
                    try:
                        # Try to parse as JSON dictionary
                        mission_dict = json.loads(match)
                        if isinstance(mission_dict, dict):
                            # Add single quotes to string values
                            for key, value in mission_dict.items():
                                if isinstance(value, str):
                                    mission_dict[key] = f"'{value}'"
                            return mission_dict
                    except json.JSONDecodeError:
                        continue
            print("No valid dictionary found in the matches.")
            return {}
        else:
            print("No dictionary-like pattern found in the string.")
            return {}
    except ValueError as e:
        print(f"Error converting string to dictionary: {e}")
        return {}
    except SyntaxError as e:
        print(f"Syntax error in the string: {e}")
        return {}
    

def find_character_position(game_str, character):
    # Split the game_str into lines
    lines = game_str.split('\n')
    
    # Search for the character in each line
    for x, line in enumerate(lines):
        if character in line:
            y = line.index(character)
            return (x, y)  # Return as soon as the character is found

    return None  # Return None if the character is not found

def string_to_underscores(input_string):
    return input_string.replace(" ", "_")

def update_csv(file_path, row_data):
    # Check if the CSV file exists
    if os.path.exists(file_path):
        # Load the existing CSV file
        df = pd.read_csv(file_path)
        
        # Ensure the columns 'file_name' and 'description' exist
        if 'file_name' not in df.columns or 'description' not in df.columns:
            df['file_name'] = df.get('file_name', pd.Series())
            df['description'] = df.get('description', pd.Series())
    else:
        # Create a new DataFrame with the specified columns if the file does not exist
        df = pd.DataFrame(columns=['file_name', 'description'])
    
    # Append the new row to the DataFrame
    new_row_df = pd.DataFrame([row_data], columns=['file_name', 'description'])
    df = pd.concat([df, new_row_df], ignore_index=True)
    
    # Save the DataFrame to the CSV file
    df.to_csv(file_path, index=False)

def diff_dict(A, B):
    # Return a dictionary that contains only the keys from B that are not in A
    return {key: B[key] for key in B if key not in A}

def overlap_dict(dict_A, dict_B):
    result = {}
    for key_A, value_A in dict_A.items():
        if key_A in dict_B.values():
            result[key_A] = value_A
    return result

def simple_similarity(desc1: str, desc2: str) -> int:
    """
    Calculate a simple similarity score based on the number of common words.
    
    Parameters:
    desc1 (str): First description.
    desc2 (str): Second description.
    
    Returns:
    int: Count of common words.
    """
    words1 = set(desc1.lower().split())
    words2 = set(desc2.lower().split())
    return len(words1.intersection(words2))

def bert_similarity(desc1: str, desc2: str) -> float:
    """
    Calculate similarity using BERT embeddings and cosine similarity.

    Parameters:
    desc1 (str): First description.
    desc2 (str): Second description.

    Returns:
    float: Cosine similarity score.
    """
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained('bert-base-uncased')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on {device}!")

    tokens1 = tokenizer(desc1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    tokens2 = tokenizer(desc2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        embedding1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embedding2 = model(**tokens2).last_hidden_state.mean(dim=1)
    similarity = 1 - cosine(embedding1[0].cpu().numpy(), embedding2[0].cpu().numpy())
    print(f"Similarity between {desc1} and {desc2} is {similarity}")
    return similarity

def bert_batch_similarity(descs1, descs2):
    """
    Calculate similarities for batches of descriptions using DistilBERT embeddings and cosine similarity.

    Parameters:
    descs1 (List[str]): First list of descriptions.
    descs2 (List[str]): Second list of descriptions.

    Returns:
    List[float]: List of cosine similarity scores.
    """

    #tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #model = AutoModel.from_pretrained('bert-base-uncased')

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Tokenize and encode the batches of descriptions
    tokens1 = tokenizer(descs1, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    tokens2 = tokenizer(descs2, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)

    with torch.no_grad():
        embedding1 = model(**tokens1).last_hidden_state.mean(dim=1)
        embedding2 = model(**tokens2).last_hidden_state.mean(dim=1)

    # Compute cosine similarities
    similarities = [1 - cosine(e1.cpu().numpy(), e2.cpu().numpy()) for e1, e2 in zip(embedding1, embedding2)]

    return similarities

def find_most_similar_images(dictionary, csv_path):
    """
    Find the most semantically similar image for each tile in the dictionary.

    Parameters:
    dictionary (Dict[str, str]): A dictionary with descriptions as keys and tiles as values.
    csv_path (str): Path to the CSV file containing image filenames and descriptions.

    Returns:
    Tuple[Dict[str, Image.Image], Dict[str, int]]: A tuple containing two dictionaries,
        one with the tiles and corresponding Image objects, and the other with tiles and similarity scores.
    """
    # Load the CSV file
    folder = f'{csv_path}/world_tileset_data'
    folder_char = f'{csv_path}/character_sprite_data'
    data = pd.read_csv(f"{folder}/metadata.csv")
    data_char = pd.read_csv(f"{folder_char}/metadata.csv")
    # Initialize dictionaries for image objects and similarity scores
    images = {}
    similarity_scores = {}

    # Iterate over the dictionary to find the most similar image for each tile
    for desc, tile in dictionary.items():
        if not tile.isalpha():
            max_similarity = 0
            selected_image = None

            # Compare each description with the descriptions in the CSV
            #for _, row in data.iterrows():
            #    similarity = bert_similarity(desc, row['description'])
            #    if similarity >= max_similarity:
            #        max_similarity = similarity
            #        selected_image = row['filename']

            similarities = bert_batch_similarity([desc] * len(list(data_char['description'])), list(data_char['description']))
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            selected_image = data_char.iloc[max_index]['filename']

            # Load the image and store it along with the similarity score
            if selected_image:
                #image_path = os.path.join(os.path.dirname(csv_path), selected_image)
                image_path = f"{folder_char}/{selected_image}"
                images[tile] = Image.open(image_path).convert("RGBA")
                similarity_scores[tile] = max_similarity
        else:
            max_similarity = 0
            selected_image = None

            # Compare each description with the descriptions in the CSV
            similarities = bert_batch_similarity([desc] * len(list(data['description'])), list(data['description']))
            max_similarity = max(similarities)
            max_index = similarities.index(max_similarity)
            selected_image = data.iloc[max_index]['filename']

            # Load the image and store it along with the similarity score
            if selected_image:
                #image_path = os.path.join(os.path.dirname(csv_path), selected_image)
                image_path = f"{folder}/{selected_image}"
                images[tile] = Image.open(image_path).convert("RGBA")
                similarity_scores[tile] = max_similarity

    return images, similarity_scores

def find_most_similar_images_gpt(dictionary, csv_path, openai_function, model):
    """
    Find the most semantically similar image for each tile in the dictionary.

    Parameters:
    dictionary (Dict[str, str]): A dictionary with descriptions as keys and tiles as values.
    csv_path (str): Path to the CSV file containing image filenames and descriptions.

    Returns:
    Tuple[Dict[str, Image.Image], Dict[str, int]]: A tuple containing two dictionaries,
        one with the tiles and corresponding Image objects, and the other with tiles and similarity scores.
    """
    # Load the CSV file
    folder = f'{csv_path}/world_tileset_data'
    folder_char = f'{csv_path}/character_sprite_data'
    df = pd.read_csv(f"{folder}/metadata.csv")
    df_char = pd.read_csv(f"{folder_char}/metadata.csv")


    # Initialize dictionaries for image objects and similarity scores
    images = {}

    # Iterate over the dictionary to find the most similar image for each tile
    for desc, tile in dictionary.items():
        max_similarity = 0
        selected_image = None
        output_cost = []
        input_cost = []

        if not desc.isalpha():
            # Compare each description with the descriptions in the CSV
            NO_OF_EXCEPTIONS = 0
            done = False
            try:
                while not done:    
                    similarity_discriptions = openai_function(model="gpt-4-0613", messages=[
                                                                                    {"role": "system", "content": "You are a great sementic similarity checker. You can check for semantic similarities and return without hillucinating. Do not return None. There will always be a word similar to presented description"},
                                                                                    {"role": "user", "content": f"out of the following list of description:\n{list(df_char.description)}\nwhich one description matches '{desc}'. Strictly return only the one word that matches the most. Only and only the word that matches. Do not return None. There will always be a word similar to presented description"}
                                                                                    ], 
                                                                                    temperature = 0.6)
                    
                    print(f"Similar to {desc} is {str(similarity_discriptions['choices'][0]['message']['content'])}")
                    if str(similarity_discriptions['choices'][0]['message']['content']) == 'None':
                        NO_OF_EXCEPTIONS += 1
                        if NO_OF_EXCEPTIONS >= 5:
                            done = True
                        continue
                    selected_image = df_char.loc[df_char['description'] == str(similarity_discriptions['choices'][0]['message']['content']), 'filename'].values[0]
                    input_cost.append(similarity_discriptions["usage"]["prompt_tokens"])
                    output_cost.append(similarity_discriptions["usage"]["completion_tokens"])
                    
                    if selected_image:
                        #image_path = os.path.join(os.path.dirname(folder_char), selected_image)
                        image_path = f"{folder_char}/{selected_image}"
                        images[tile] = Image.open(image_path).convert("RGBA")
                    done=True
            except Exception as e:
                #print(f"check#3 done = {done}")
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n {tb}")
                NO_OF_EXCEPTIONS += 1
                if NO_OF_EXCEPTIONS >= 5:
                    done = True
                pass
        else:
            # Compare each description with the descriptions in the CSV
            NO_OF_EXCEPTIONS = 0
            done = False
            try:
                while not done:   
                    similarity_discriptions = openai_function(model="gpt-4-0613", messages=[
                                                                                    {"role": "system", "content": "You are a great sementic similarity checker. You can check for semantic similarities and return without hillucinating. Do not return None. There will always be a word similar to presented description"},
                                                                                    {"role": "user", "content": f"out of the following list of description:\n{list(df.description)}\nwhich one description matches '{desc}'. Strictly return only the one word that matches the most. Only and only the word that matches. Do not return None. There will always be a word similar to presented description"}
                                                                                    ], 
                                                                                    temperature = 0.6)
                    
                    
                    print(f"Similar to {desc} is {str(similarity_discriptions['choices'][0]['message']['content'])}")
                    if str(similarity_discriptions['choices'][0]['message']['content']) == 'None':
                        NO_OF_EXCEPTIONS += 1
                        if NO_OF_EXCEPTIONS >= 5:
                            done = True
                        continue
                    selected_image = df.loc[df['description'] == str(similarity_discriptions['choices'][0]['message']['content']), 'filename'].values[0]
                    input_cost.append(similarity_discriptions["usage"]["prompt_tokens"])
                    output_cost.append(similarity_discriptions["usage"]["completion_tokens"])
                    
                        
                    # Load the image and store it along with the similarity score
                    if selected_image:
                        #image_path = os.path.join(os.path.dirname(folder), selected_image)
                        image_path = f"{folder}/{selected_image}"
                        images[tile] = Image.open(image_path).convert("RGBA")
                    done=True

            except Exception as e:
                #print(f"check#3 done = {done}")
                tb = traceback.format_exc()
                print(f"Exception raised: {e}\n {tb}")
                NO_OF_EXCEPTIONS += 1
                if NO_OF_EXCEPTIONS >= 5:
                    done = True
                pass

    return images, sum(input_cost), sum(output_cost)

def scale_string(s, scale_factor):
    scaled_lines = []
    # Define special characters
    special_chars = "@#$%^&*()-+=[]{};:'\"\\|,.<>/?!"

    # Split the string into lines
    lines = s.strip().split('\n')

    # Helper function to find the nearest non-special character in the line
    def find_nearest_alphabet(line, index):
        left = right = index
        while left >= 0 or right < len(line):
            if left >= 0 and line[left] not in special_chars:
                return line[left]
            if right < len(line) and line[right] not in special_chars:
                return line[right]
            left -= 1
            right += 1
        return ' '  # Default to space if no nearby character is found

    for line in lines:
        new_line = ''
        for i, char in enumerate(line):
            if char in special_chars:
                nearest_char = find_nearest_alphabet(line, i)
                new_line += nearest_char * (scale_factor - 1) + char
            else:
                new_line += char * scale_factor

        # Scale the line vertically
        for i in range(scale_factor):
            if i == 0:
                scaled_lines.append(new_line)
            else:
                # For lines other than the first, replace special characters with nearest characters
                modified_line = ''
                for char in new_line:
                    if char in special_chars:
                        modified_line += nearest_char
                    else:
                        modified_line += char
                scaled_lines.append(modified_line)

    return '\n'.join(scaled_lines)
