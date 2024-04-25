import gym
from gym import spaces
import numpy as np
from PIL import Image

import os
import json
import time
from utils import map_to_list, load_image_dict
from solvers import find_characters
import imageio
from fixers import pad_rows_to_max_length
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

class StoryWorldEnv(gym.Env):
    def __init__(self, walkable_tiles,tiles_without_char,  tiles, str_map_without_chars, str_map, interactive_object_tiles, enemy_tiles):
        super(StoryWorldEnv, self).__init__()
        str_map = pad_rows_to_max_length(str_map)
        str_map_without_chars = pad_rows_to_max_length(str_map_without_chars)
        
        self.map_str_without_chars = str_map_without_chars.strip().split('\n')
        self.map_str = str_map.strip().split('\n')
        
        self.tile_size = 16
        self.char_tile_size = 16
        self.tiles = tiles
        self.tiles_without_char = tiles_without_char
        self.action_space = spaces.Discrete(6)  # Up, down, left, right, pick, hit
        self.observation_space = spaces.Box(low=0, high=255, shape=(len(self.map_str) * self.tile_size, len(self.map_str[0]) * self.tile_size, 3), dtype=np.uint8)
        self.default_walkable_tile = "B"
        
        self.walkable_tiles = walkable_tiles
        self.interactive_object_tiles = interactive_object_tiles
        self.enemy_tiles = enemy_tiles
        self.picked_objects = []

        # Make second layer transparent

        for char, image in self.tiles.items():
            if char.isalpha():
                self.tiles[char] = remove(self.tiles[char])


        # Count the occurrences of each tile in the map
        tile_counts = {}
        for row in self.map_str:
            for tile in row:
                if tile in walkable_tiles:
                    if tile not in tile_counts:
                        tile_counts[tile] = 1
                    else:
                        tile_counts[tile] += 1

        # Determine the most common walkable tile
        if tile_counts:
            self.default_walkable_tile = max(tile_counts, key=tile_counts.get)
        else:
            raise ValueError("No walkable tiles found in the map.")
        

        self.reset()

    def reset(self):
        self.map = [list(row) for row in self.map_str]
        self.map_without_chars = [list(row) for row in self.map_str_without_chars]
        self.grid_width = max(len(row) for row in self.map)
        self.grid_height = len(self.map)
        self.player_pos = self.find_player_position()
        self.current_tile = self.default_walkable_tile  # Default current tile to 'A', change if necessary
        return self.get_state()

    def step(self, action):
        reward = 0
        if action < 4:  # Movement actions
            self.move_player(action)
        elif action == 4:  # Pick action
            reward += self.pick_object()
        elif action == 5:  # Hit action
            reward += self.hit_enemy()

        
        done = False
        info = {}
        return self.get_state(), reward, done, info

    def move_player(self, action):
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
        dx, dy = moves[action]
        new_row = self.player_pos[0] + dx
        new_col = self.player_pos[1] + dy

        if 0 <= new_row < len(self.map) and 0 <= new_col < len(self.map[0]):
            new_tile = self.map[new_row][new_col]
            if new_tile in self.walkable_tiles:
                self.update_player_position(new_row, new_col, new_tile)

    def update_player_position(self, new_row, new_col, new_tile):
        self.map[self.player_pos[0]][self.player_pos[1]] = self.current_tile
        self.player_pos = (new_row, new_col)
        self.current_tile = new_tile
        self.map[new_row][new_col] = '@'

    def pick_object(self):
        reward = 0
        # Check adjacent tiles for interactive objects and pick them if present
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.interactive_object_tiles:
                    print("Picked an object!")
                    self.map[new_y][new_x] = self.default_walkable_tile 
                    reward = 1
                    break  # Exit after picking up one object
        return reward

    def hit_enemy(self):
        reward = 0
        # Check adjacent tiles for enemies and hit them if present
        adjacent_positions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
        for dx, dy in adjacent_positions:
            x, y = self.player_pos
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height:
                target_tile = self.map[new_y][new_x]
                if target_tile in self.enemy_tiles:  # Assuming enemy_tiles is a list of enemy tile identifiers
                    print("Hit an enemy!")
                    self.map[new_y][new_x] = self.default_walkable_tile  # Replace with default or empty tile
                    reward = 5
                    break  # Exit after hitting one enemy
        return reward
    def get_state(self):
        #print(self.map)
        row_lengths = [len(row) for row in self.map]
        assert len(set(row_lengths)) == 1, "Not all rows in the map have the same length"
        return np.array(self.map)

    def render(self, mode='human'):
        env_img = Image.new('RGBA', (len(self.map[0]) * self.tile_size, len(self.map) * self.tile_size))
        
        # 1st layer
        for i, row in enumerate(self.map_without_chars):
            for j, tile in enumerate(row):
                tile_img = self.tiles_without_char[self.default_walkable_tile].resize((self.tile_size, self.tile_size))
                # Use the image itself as the mask to handle transparency
                env_img.paste(tile_img, (j * self.tile_size, i * self.tile_size), tile_img)

        ## 2nd layer
        for i, row in enumerate(self.map_without_chars):
            for j, tile in enumerate(row):
                tile_img = self.tiles_without_char[tile].resize((self.tile_size, self.tile_size))
                # Use the image itself as the mask to handle transparency
                env_img.paste(tile_img, (j * self.tile_size, i * self.tile_size), tile_img)
        # 3rd layer
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if not tile.isalpha():
                    tile_img = self.tiles[tile].resize((self.char_tile_size, self.char_tile_size))
                    # Use the image itself as the mask to handle transparency
                    env_img.paste(tile_img, (j * self.char_tile_size, i * self.char_tile_size), tile_img)    
                else:
                    tile_img = self.tiles[tile].resize((self.tile_size, self.tile_size))
                    # Use the image itself as the mask to handle transparency
                    env_img.paste(tile_img, (j * self.tile_size, i * self.tile_size), tile_img)
         # Draw the picked objects and their count
        if mode == 'human' or mode == 'rgb_array':
            draw = ImageDraw.Draw(env_img)
            font = ImageFont.load_default()
            text = f"Objects Picked: {len(self.picked_objects)}"
            text_position = (10, env_img.size[1] - 10)  # Position at bottom left corner
            draw.text(text_position, text, (255, 255, 255), font=font)

            # Optionally, draw the icons of the picked objects next to the text
            x_offset = 150  # Starting x position to draw picked objects
            for obj in self.picked_objects:
                obj_img = self.tiles[obj].resize((self.tile_size // 2, self.tile_size // 2))  # Smaller icon size
                env_img.paste(obj_img, (x_offset, env_img.size[1] - self.tile_size // 2 - 10), obj_img)
                x_offset += self.tile_size // 2 + 5  # Move to the right for the next icon


        if mode == 'human':
            # If showing to a human, convert to RGB since most viewers don't handle RGBA
            env_img.convert('RGB').show()
        elif mode == 'rgb_array':
            return np.array(env_img.convert('RGB'))  # Convert to RGB array
        elif mode == 'image':
            return env_img

    def find_player_position(self):
        for i, row in enumerate(self.map):
            for j, tile in enumerate(row):
                if tile == '@':
                    return (i, j)
        return None

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        
        return self.action_space.sample()
    
class LLMAgent:
    def __init__(self):
        pass

    def action(self, action_string):
        if action_string == 'move_up':
            return 0
        if action_string == 'move_down':
            return 1
        if action_string == 'move_left':
            return 2
        if action_string == 'move_right':
            return 3
        if action_string == 'pick_object':
            return 4
        if action_string == 'hit_enemy':
            return 5