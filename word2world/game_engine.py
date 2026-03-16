import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageColor, ImageDraw, ImageFont

from word2world.fixers import pad_rows_to_max_length
from word2world.solvers import find_characters

Direction = Tuple[int, int]

DIRECTION_OFFSETS: Dict[str, Direction] = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}


def load_game_data(game_path: str) -> dict:
    with open(game_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _grid_to_rows(text: str) -> List[List[str]]:
    padded = pad_rows_to_max_length(text)
    return [list(row) for row in padded.splitlines()]


def _rows_to_text(rows: List[List[str]]) -> str:
    return "\n".join("".join(row) for row in rows)


def _extract_allowed_chars(tile_mapping: Dict[str, str], rows: Iterable[List[str]]) -> set:
    allowed = set(tile_mapping.values())
    for row in rows:
        allowed.update(row)
    return allowed


def _parse_tile_list(raw_value, allowed_chars: set) -> set:
    if isinstance(raw_value, list):
        return {value for value in raw_value if isinstance(value, str) and len(value) == 1}

    if not isinstance(raw_value, str):
        return set()

    try:
        parsed = ast.literal_eval(raw_value)
        if isinstance(parsed, list):
            parsed_values = set()
            for value in parsed:
                if isinstance(value, str):
                    if len(value) == 1:
                        parsed_values.add(value)
                    else:
                        parsed_values.update(char for char in value if char in allowed_chars)
            if parsed_values:
                return parsed_values
    except (SyntaxError, ValueError):
        pass

    parsed_values = {match[1] for match in __import__("re").findall(r"(['\"])(.)\1", raw_value)}
    special_values = {char for char in allowed_chars if not char.isalnum() and char in raw_value}
    return parsed_values.union(special_values)


def _most_common_walkable_tile(rows: List[List[str]], walkables: set) -> str:
    counts: Dict[str, int] = {}
    for row in rows:
        for tile in row:
            if tile in walkables:
                counts[tile] = counts.get(tile, 0) + 1

    if counts:
        return max(counts, key=counts.get)

    for row in rows:
        if row:
            return row[0]

    return "."


def _asset_files_exist(tile_data_dir: str) -> bool:
    base_path = Path(tile_data_dir)
    if not base_path.exists():
        return False

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    for folder_name in ("world_tileset_data", "character_sprite_data"):
        folder = base_path / folder_name
        if not folder.exists():
            continue
        for pattern in patterns:
            if any(folder.glob(pattern)):
                return True
    return False


@lru_cache(maxsize=8)
def _load_tileset_images_cached(mapping_items: Tuple[Tuple[str, str], ...], tile_data_dir: str) -> Dict[str, Image.Image]:
    if not tile_data_dir or not _asset_files_exist(tile_data_dir):
        return {}

    try:
        from word2world.utils import find_most_similar_images

        images, _ = find_most_similar_images(dict(mapping_items), tile_data_dir)
        return {key: value.convert("RGBA") for key, value in images.items()}
    except Exception:
        return {}


def load_tileset_images(tile_mapping: Dict[str, str], tile_data_dir: str) -> Dict[str, Image.Image]:
    return _load_tileset_images_cached(tuple(sorted(tile_mapping.items())), tile_data_dir)


def _fallback_color_for_char(tile_char: str) -> Tuple[int, int, int]:
    digest = hashlib.md5(tile_char.encode("utf-8")).hexdigest()
    rgb_hex = f"#{digest[:6]}"
    return ImageColor.getrgb(rgb_hex)


@lru_cache(maxsize=256)
def _build_fallback_tile(tile_char: str, tile_size: int) -> Image.Image:
    image = Image.new("RGBA", (tile_size, tile_size), _fallback_color_for_char(tile_char))
    draw = ImageDraw.Draw(image)
    border_color = (255, 255, 255, 160) if tile_char.isalnum() else (0, 0, 0, 180)
    draw.rectangle((0, 0, tile_size - 1, tile_size - 1), outline=border_color)

    if tile_char:
        font = ImageFont.load_default()
        text_color = (0, 0, 0, 255) if tile_char.isupper() else (255, 255, 255, 255)
        bbox = draw.textbbox((0, 0), tile_char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x_pos = max(0, (tile_size - text_width) // 2)
        y_pos = max(0, (tile_size - text_height) // 2)
        draw.text((x_pos, y_pos), tile_char, fill=text_color, font=font)

    return image


class Word2WorldGame:
    def __init__(
        self,
        data: dict,
        round_key: str = "round_0",
        tile_data_dir: str = "word2world/data",
        camera_width: int = 20,
        camera_height: int = 16,
        info_panel_height: int = 4,
    ):
        if round_key not in data:
            raise KeyError(f"Round '{round_key}' does not exist in the supplied game data.")

        self.data = data
        self.round_key = round_key
        self.round_data = data[round_key]
        self.tile_data_dir = tile_data_dir
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.info_panel_height = info_panel_height

        self.story = self.round_data.get("story", "")
        self.goals = self.round_data.get("goals", "")
        self.objectives = self.round_data.get("objectives", {})
        self.evaluations = self.round_data.get("evaluations", {})
        self.complexity = self.round_data.get("complexity", {})
        self.tile_mapping = self.round_data.get("tile_mapping", {})

        self.grid_world = _grid_to_rows(self.round_data["world"])
        self.grid_first_layer = _grid_to_rows(self.round_data["world_1st_layer"]["world"])

        allowed_chars = _extract_allowed_chars(self.tile_mapping, self.grid_world + self.grid_first_layer)
        self.walkables = _parse_tile_list(self.round_data.get("walkable_tiles", []), allowed_chars)
        self.important_tiles = _parse_tile_list(self.round_data.get("important_tiles", []), allowed_chars)
        self.interactive_object_tiles = _parse_tile_list(
            self.round_data.get("interactive_object_tiles", []), allowed_chars
        )

        self.tileset = load_tileset_images(self.tile_mapping, tile_data_dir)
        self.default_walkable_tile = _most_common_walkable_tile(self.grid_first_layer, self.walkables)

        world_text = _rows_to_text(self.grid_world)
        characters = find_characters(world_text)
        if "@" not in characters:
            raise ValueError("The selected round does not contain a player '@' tile.")

        self.player_pos = [characters["@"][0], characters["@"][1]]
        enemy_position = characters.get("#", (-1, -1))
        self.enemy_pos = [enemy_position[0], enemy_position[1]]
        self.initial_enemy_x = self.enemy_pos[0]
        self.enemy_direction = 1

        self.camera_pos = [0, 0]
        self.last_direction: Direction = DIRECTION_OFFSETS["right"]
        self.picked_objects: Dict[str, int] = {}
        self.enemy_bullets: List[List[int]] = []
        self.player_bullets: List[List[int]] = []
        self.messages: List[str] = []
        self.running = True
        self.player_won = False
        self.player_hit = False

        self.update_camera()

    @property
    def width(self) -> int:
        return len(self.grid_world[0]) if self.grid_world else 0

    @property
    def height(self) -> int:
        return len(self.grid_world)

    @property
    def enemy_alive(self) -> bool:
        return self.enemy_pos[0] != -1 and self.enemy_pos[1] != -1

    def _record(self, message: str) -> None:
        self.messages.append(message)
        self.messages = self.messages[-10:]

    def _normalize_direction(self, direction: Optional[object]) -> Direction:
        if direction is None:
            return self.last_direction

        if isinstance(direction, str):
            if direction not in DIRECTION_OFFSETS:
                raise ValueError(f"Unknown direction '{direction}'.")
            return DIRECTION_OFFSETS[direction]

        if isinstance(direction, tuple) and len(direction) == 2:
            return direction

        if isinstance(direction, list) and len(direction) == 2:
            return direction[0], direction[1]

        raise ValueError(f"Unsupported direction value: {direction}")

    def update_camera(self) -> None:
        max_x = max(0, self.width - self.camera_width)
        max_y = max(0, self.height - self.camera_height)
        self.camera_pos[0] = max(0, min(self.player_pos[0] - self.camera_width // 2, max_x))
        self.camera_pos[1] = max(0, min(self.player_pos[1] - self.camera_height // 2, max_y))

    def move_player(self, dx: int, dy: int) -> bool:
        if not self.running:
            return False

        self.last_direction = (dx, dy)
        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy

        if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
            self._record("You cannot move outside the world.")
            return False

        if self.grid_world[new_y][new_x] not in self.walkables:
            self._record("That tile is blocked.")
            return False

        self.player_pos[0] = new_x
        self.player_pos[1] = new_y
        self.update_camera()
        self.pick_object()
        return True

    def pick_object(self) -> None:
        x_pos, y_pos = self.player_pos
        target_tile = self.grid_world[y_pos][x_pos]

        if target_tile in self.interactive_object_tiles:
            self.grid_world[y_pos][x_pos] = self.default_walkable_tile
            self.picked_objects[target_tile] = self.picked_objects.get(target_tile, 0) + 1
            self._record(f"Picked up tile '{target_tile}'.")

    def move_enemy(self) -> None:
        if not self.enemy_alive:
            return

        new_x = self.enemy_pos[0] + self.enemy_direction
        if abs(new_x - self.initial_enemy_x) > 5:
            self.enemy_direction *= -1
            return

        if new_x < 0 or new_x >= self.width or self.grid_world[self.enemy_pos[1]][new_x] not in self.walkables:
            self.enemy_direction *= -1
            return

        self.enemy_pos[0] = new_x

    def enemy_detect_player(self) -> bool:
        if not self.enemy_alive:
            return False

        if abs(self.player_pos[0] - self.enemy_pos[0]) <= 3 and self.player_pos[1] == self.enemy_pos[1]:
            return True
        if abs(self.player_pos[1] - self.enemy_pos[1]) <= 3 and self.player_pos[0] == self.enemy_pos[0]:
            return True
        return False

    def enemy_attack_player(self) -> None:
        if not self.enemy_alive:
            return

        if self.player_pos[0] < self.enemy_pos[0]:
            self.enemy_bullets.append([self.enemy_pos[0], self.enemy_pos[1], -1, 0])
        elif self.player_pos[0] > self.enemy_pos[0]:
            self.enemy_bullets.append([self.enemy_pos[0], self.enemy_pos[1], 1, 0])
        elif self.player_pos[1] < self.enemy_pos[1]:
            self.enemy_bullets.append([self.enemy_pos[0], self.enemy_pos[1], 0, -1])
        elif self.player_pos[1] > self.enemy_pos[1]:
            self.enemy_bullets.append([self.enemy_pos[0], self.enemy_pos[1], 0, 1])

    def player_shoot(self, direction: Optional[object] = None) -> None:
        if not self.running:
            return

        dx, dy = self._normalize_direction(direction)
        self.last_direction = (dx, dy)
        self.player_bullets.append([self.player_pos[0], self.player_pos[1], dx, dy])
        self._record("You fired a shot.")

    def hit_enemy(self, direction: Optional[object] = None) -> bool:
        if not self.enemy_alive:
            self._record("No enemy is active in this round.")
            return False

        dx, dy = self._normalize_direction(direction)
        self.last_direction = (dx, dy)
        x_pos = self.player_pos[0] + dx
        y_pos = self.player_pos[1] + dy

        if 0 <= x_pos < self.width and 0 <= y_pos < self.height and (x_pos, y_pos) == tuple(self.enemy_pos):
            self.enemy_pos = [-1, -1]
            self.player_won = True
            self._record("Enemy defeated.")
            return True

        self._record("Your attack did not connect.")
        return False

    def move_bullets(self) -> None:
        for bullet in self.enemy_bullets[:]:
            bullet[0] += bullet[2]
            bullet[1] += bullet[3]

            if bullet[0] == self.player_pos[0] and bullet[1] == self.player_pos[1]:
                self.enemy_bullets.remove(bullet)
                self.running = False
                self.player_hit = True
                self._record("Player hit.")
                continue

            if (
                bullet[0] < 0
                or bullet[0] >= self.width
                or bullet[1] < 0
                or bullet[1] >= self.height
                or self.grid_world[bullet[1]][bullet[0]] not in self.walkables
            ):
                self.enemy_bullets.remove(bullet)

        for bullet in self.player_bullets[:]:
            bullet[0] += bullet[2]
            bullet[1] += bullet[3]

            if self.enemy_alive and bullet[0] == self.enemy_pos[0] and bullet[1] == self.enemy_pos[1]:
                self.enemy_pos = [-1, -1]
                self.player_won = True
                self.player_bullets.remove(bullet)
                self._record("Enemy hit.")
                continue

            if (
                bullet[0] < 0
                or bullet[0] >= self.width
                or bullet[1] < 0
                or bullet[1] >= self.height
                or self.grid_world[bullet[1]][bullet[0]] not in self.walkables
            ):
                self.player_bullets.remove(bullet)

    def advance_world(self) -> None:
        if not self.running:
            return

        if self.enemy_alive:
            if self.enemy_detect_player():
                self.enemy_attack_player()
            else:
                self.move_enemy()

        self.move_bullets()

    def step(self, action: str, direction: Optional[object] = None) -> None:
        if action == "move":
            dx, dy = self._normalize_direction(direction)
            self.move_player(dx, dy)
        elif action == "shoot":
            self.player_shoot(direction)
        elif action == "melee":
            self.hit_enemy(direction)
        elif action == "wait":
            self._record("You wait for a moment.")
        else:
            raise ValueError(f"Unsupported action '{action}'.")

        self.advance_world()

    def get_tile_image(self, tile_char: str, tile_size: int) -> Image.Image:
        image = self.tileset.get(tile_char)
        if image is None:
            return _build_fallback_tile(tile_char, tile_size)
        return image.resize((tile_size, tile_size), Image.Resampling.NEAREST)

    def render_frame(self, tile_size: int = 16) -> Image.Image:
        width = self.camera_width * tile_size
        height = (self.camera_height + self.info_panel_height) * tile_size
        image = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        draw = ImageDraw.Draw(image)

        for y_pos in range(self.camera_height):
            for x_pos in range(self.camera_width):
                world_x = x_pos + self.camera_pos[0]
                world_y = y_pos + self.camera_pos[1]

                if world_y >= len(self.grid_first_layer) or world_x >= len(self.grid_first_layer[0]):
                    continue

                origin_x = x_pos * tile_size
                origin_y = y_pos * tile_size

                base_tile = self.get_tile_image(self.default_walkable_tile, tile_size)
                image.alpha_composite(base_tile, (origin_x, origin_y))

                first_layer_tile = self.grid_first_layer[world_y][world_x]
                if first_layer_tile not in {"@", "#"}:
                    image.alpha_composite(self.get_tile_image(first_layer_tile, tile_size), (origin_x, origin_y))

                world_tile = self.grid_world[world_y][world_x]
                if world_tile not in {"@", "#"}:
                    image.alpha_composite(self.get_tile_image(world_tile, tile_size), (origin_x, origin_y))

        if self.enemy_alive:
            enemy_x = (self.enemy_pos[0] - self.camera_pos[0]) * tile_size
            enemy_y = (self.enemy_pos[1] - self.camera_pos[1]) * tile_size
            if 0 <= enemy_x < width and 0 <= enemy_y < self.camera_height * tile_size:
                image.alpha_composite(self.get_tile_image("#", tile_size), (enemy_x, enemy_y))

        player_x = (self.player_pos[0] - self.camera_pos[0]) * tile_size
        player_y = (self.player_pos[1] - self.camera_pos[1]) * tile_size
        if 0 <= player_x < width and 0 <= player_y < self.camera_height * tile_size:
            image.alpha_composite(self.get_tile_image("@", tile_size), (player_x, player_y))

        for bullet in self.enemy_bullets:
            bullet_x = (bullet[0] - self.camera_pos[0]) * tile_size + tile_size // 2
            bullet_y = (bullet[1] - self.camera_pos[1]) * tile_size + tile_size // 2
            if 0 <= bullet_x < width and 0 <= bullet_y < self.camera_height * tile_size:
                radius = max(2, tile_size // 6)
                draw.ellipse(
                    (bullet_x - radius, bullet_y - radius, bullet_x + radius, bullet_y + radius),
                    fill=(255, 255, 255, 255),
                )

        for bullet in self.player_bullets:
            bullet_x = (bullet[0] - self.camera_pos[0]) * tile_size + tile_size // 2
            bullet_y = (bullet[1] - self.camera_pos[1]) * tile_size + tile_size // 2
            if 0 <= bullet_x < width and 0 <= bullet_y < self.camera_height * tile_size:
                radius = max(2, tile_size // 6)
                draw.ellipse(
                    (bullet_x - radius, bullet_y - radius, bullet_x + radius, bullet_y + radius),
                    fill=(255, 165, 0, 255),
                )

        panel_top = self.camera_height * tile_size
        draw.rectangle((0, panel_top, width, height), fill=(0, 0, 0, 255))
        font = ImageFont.load_default()
        cursor_x = 0
        for tile_char, count in self.picked_objects.items():
            image.alpha_composite(self.get_tile_image(tile_char, tile_size), (cursor_x, panel_top))
            draw.text(
                (cursor_x, panel_top + tile_size),
                str(count),
                fill=(255, 255, 255, 255),
                font=font,
            )
            cursor_x += tile_size * 2

        return image
