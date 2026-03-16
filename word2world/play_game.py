import argparse
import os
import sys

import pygame

from word2world.configs import Config
from word2world.game_engine import DIRECTION_OFFSETS, Word2WorldGame, load_game_data

TILE_SIZE = 16
CAMERA_WIDTH = 20
CAMERA_HEIGHT = 16
INFO_PANEL_HEIGHT = 4

KEY_TO_DIRECTION = {
    pygame.K_a: "left",
    pygame.K_d: "right",
    pygame.K_w: "up",
    pygame.K_s: "down",
}


def pil_to_pygame(pil_image):
    mode = pil_image.mode
    size = pil_image.size
    data = pil_image.tobytes()
    return pygame.image.fromstring(data, size, mode).convert_alpha()


def parse_args():
    parser = argparse.ArgumentParser(description="Play a Word2World game with pygame.")
    parser.add_argument(
        "--game_path",
        type=str,
        help="Path to a generated game JSON file. Defaults to word2world/examples/example_1.json",
    )
    parser.add_argument(
        "--round_number",
        type=str,
        default="round_0",
        help="Round key to play, for example round_0 or round_1.",
    )
    return parser.parse_args()


def load_selected_game(game_path: str = None):
    if game_path:
        if not os.path.exists(game_path):
            raise ValueError(f"{game_path} does not exist. Please provide an existing path.")
        return load_game_data(game_path)

    default_path = os.path.join("word2world", "examples", "example_1.json")
    return load_game_data(default_path)


def main():
    args = parse_args()
    cfg = Config()
    data = load_selected_game(args.game_path)

    game = Word2WorldGame(
        data=data,
        round_key=args.round_number,
        tile_data_dir=cfg.tile_data_dir,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        info_panel_height=INFO_PANEL_HEIGHT,
    )

    pygame.init()
    frame = game.render_frame(tile_size=TILE_SIZE)
    screen = pygame.display.set_mode(frame.size)
    pygame.display.set_caption(f"Word2World Game - {args.round_number}")
    clock = pygame.time.Clock()

    move_direction = None
    shooting = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_DIRECTION:
                    move_direction = KEY_TO_DIRECTION[event.key]
                    game.last_direction = DIRECTION_OFFSETS[move_direction]
                elif event.key == pygame.K_SPACE:
                    game.hit_enemy(move_direction or game.last_direction)
                elif event.key == pygame.K_z:
                    shooting = True
            elif event.type == pygame.KEYUP:
                if event.key in KEY_TO_DIRECTION and KEY_TO_DIRECTION[event.key] == move_direction:
                    move_direction = None
                elif event.key == pygame.K_z:
                    shooting = False

        if move_direction:
            dx, dy = DIRECTION_OFFSETS[move_direction]
            if not game.move_player(dx, dy):
                move_direction = None

        if shooting:
            game.player_shoot(move_direction or game.last_direction)

        game.advance_world()

        if not game.running:
            running = False

        frame = game.render_frame(tile_size=TILE_SIZE)
        screen.blit(pil_to_pygame(frame), (0, 0))
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())