import hashlib
from pathlib import Path

import streamlit as st

from word2world.configs import Config
from word2world.game_engine import DIRECTION_OFFSETS, Word2WorldGame, load_game_data

TILE_SIZE = 24
EXAMPLES_DIR = Path("word2world") / "examples"


@st.cache_data(show_spinner=False)
def load_json_from_path(path: str):
    return load_game_data(path)


def round_sort_key(round_name: str):
    try:
        return int(round_name.split("_")[-1])
    except (TypeError, ValueError):
        return round_name


def ensure_game(source_id: str, round_key: str, data: dict, cfg: Config):
    needs_reset = (
        "game" not in st.session_state
        or st.session_state.get("game_source_id") != source_id
        or st.session_state.get("game_round_key") != round_key
    )

    if needs_reset:
        st.session_state.game = Word2WorldGame(data, round_key=round_key, tile_data_dir=cfg.tile_data_dir)
        st.session_state.game_source_id = source_id
        st.session_state.game_round_key = round_key

    return st.session_state.game


def set_facing(game: Word2WorldGame, facing: str):
    game.last_direction = DIRECTION_OFFSETS[facing]


def facing_name(game: Word2WorldGame) -> str:
    for name, direction in DIRECTION_OFFSETS.items():
        if tuple(game.last_direction) == tuple(direction):
            return name
    return "right"


def apply_action(game: Word2WorldGame, action: str, direction: str = None):
    if action in {"move", "shoot", "melee"}:
        game.step(action, direction)
    else:
        game.step(action)



def render_tile_mapping(game: Word2WorldGame):
    if not game.tile_mapping:
        st.write("No tile mapping found for this round.")
        return

    for tile_name, tile_char in sorted(game.tile_mapping.items(), key=lambda item: item[1]):
        sprite_col, text_col = st.columns([1, 4], vertical_alignment="center")
        with sprite_col:
            st.image(game.get_tile_image(tile_char, 32), width=32)
        with text_col:
            st.write(f"`{tile_char}`  {tile_name}")



def render_story_header(game: Word2WorldGame):
    """Render the full story and character legend above the game area."""
    if game.story:
        with st.expander("Story", expanded=True):
            st.write(game.story)

    player_desc = ""
    enemy_desc = ""
    for name, char in game.tile_mapping.items():
        if char == "@":
            player_desc = name
        elif char == "#":
            enemy_desc = name

    legend_parts = []
    if player_desc:
        legend_parts.append(f"**You** (`@`): {player_desc}")
    if enemy_desc:
        legend_parts.append(f"**Enemy** (`#`): {enemy_desc}")
    if legend_parts:
        st.markdown(" · ".join(legend_parts))


def render_sidebar(game: Word2WorldGame):
    status_label = "In progress"
    if game.player_hit:
        status_label = "Defeat"
    elif game.player_won:
        status_label = "Enemy defeated"

    st.subheader("Status")
    s1, s2 = st.columns(2)
    s1.metric("Player", f"({game.player_pos[0]}, {game.player_pos[1]})")
    if game.enemy_alive:
        s2.metric("Enemy", f"({game.enemy_pos[0]}, {game.enemy_pos[1]})")
    else:
        s2.metric("Enemy", "Defeated")
    st.caption(f"Facing: **{facing_name(game)}** · Status: **{status_label}**")

    if game.objectives:
        st.subheader("Objectives")
        for i, (name, details) in enumerate(game.objectives.items(), start=1):
            if len(details) >= 3:
                st.write(f"{i}. **{name}** — tile `{details[0]}` at ({details[1]}, {details[2]})")
            else:
                st.write(f"{i}. **{name}**")

    with st.expander("Tile Legend"):
        render_tile_mapping(game)

    st.subheader("Recent Events")
    if game.messages:
        for message in reversed(game.messages[-5:]):
            st.write(f"- {message}")
    else:
        st.write("No events yet.")


def render_controls(game: Word2WorldGame):
    st.subheader("Controls")
    current_facing = facing_name(game)
    selected_facing = st.selectbox(
        "Facing",
        options=list(DIRECTION_OFFSETS.keys()),
        index=list(DIRECTION_OFFSETS.keys()).index(current_facing),
    )
    set_facing(game, selected_facing)

    move_col1, move_col2, move_col3 = st.columns(3)
    if move_col2.button("Move Up", use_container_width=True):
        apply_action(game, "move", "up")
    if move_col1.button("Move Left", use_container_width=True):
        apply_action(game, "move", "left")
    if move_col2.button("Wait", use_container_width=True):
        apply_action(game, "wait")
    if move_col3.button("Move Right", use_container_width=True):
        apply_action(game, "move", "right")
    if move_col2.button("Move Down", use_container_width=True):
        apply_action(game, "move", "down")

    action_col1, action_col2, action_col3 = st.columns(3)
    if action_col1.button("Shoot", use_container_width=True):
        apply_action(game, "shoot", selected_facing)
    if action_col2.button("Melee", use_container_width=True):
        apply_action(game, "melee", selected_facing)
    if action_col3.button("Reset Round", use_container_width=True):
        st.session_state.game = Word2WorldGame(game.data, round_key=game.round_key, tile_data_dir=game.tile_data_dir)
        st.session_state.game_source_id = st.session_state.game_source_id
        st.session_state.game_round_key = game.round_key
        st.rerun()


def make_example_page(example_path: Path):
    """Build a page callable bound to a specific example JSON file."""

    def page():
        cfg = Config()
        data = load_json_from_path(str(example_path))
        round_keys = sorted(data.keys(), key=round_sort_key)
        file_stem = example_path.stem
        file_digest = hashlib.md5(str(example_path).encode("utf-8")).hexdigest()[:8]
        source_id = f"example:{file_stem}:{file_digest}"

        st.title(f"Word2World: {file_stem}")
        st.caption("Play a generated round in the browser.")

        with st.sidebar:
            st.header("Rounds")
            selected_round = st.radio("Select a round", options=round_keys, index=0)

        game = ensure_game(source_id, selected_round, data, cfg)

        render_story_header(game)

        if game.player_hit:
            st.error("The player has been hit. Reset the round to try again.")
        elif game.player_won:
            st.success("The enemy has been defeated.")

        game_col, controls_col, side_col = st.columns([3, 2, 2], gap="large")
        with game_col:
            st.image(game.render_frame(tile_size=TILE_SIZE), caption=selected_round)

        with controls_col:
            render_controls(game)

        with side_col:
            render_sidebar(game)

    return page


def main():
    st.set_page_config(page_title="Word2World", layout="wide")

    example_paths = sorted(EXAMPLES_DIR.glob("*.json"))

    if not example_paths:
        st.error("No example JSON files were found in word2world/examples.")
        return

    pages = [
        st.Page(
            make_example_page(path),
            title=path.stem,
            url_path=path.stem,
        )
        for path in example_paths
    ]

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()
