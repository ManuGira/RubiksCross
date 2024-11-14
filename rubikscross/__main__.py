import numpy as np

from game_app import GameApp_PyGame
from graphics import CroixPharmaGraphics
from graphics_move_functions_map import GraphicsMoveFunctionsMap
from mixer import PyGameMixer
from rubikscross import RubiksCross

TILE_SIZE = 8
TILES = [
    np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8),
    np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),
    np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),
    np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),
    np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),
    np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.uint8),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="Rubik's Cross",
        description="A kind of Rubik's Cube but in 2D and in cross shape. Controls: Press SPACE to scramble the puzzle. Use directional arrows (or WASD) to solve it. Press Q and E for 90° rotations",
    )
    parser.add_argument("-d", "--difficulty", type=int, default=2)
    import sys, os
    args=parser.parse_args(sys.argv[1:])

    assert args.difficulty < 11
    colors = np.array([
        [0, 0, 0],  # black
        [61, 204, 202],  # cyan
        [245, 100, 118],  # red
        [209, 217, 210],  # light gray
        [249, 220, 92],  # yellow
        [98, 113, 231],  # dark blue
    ], dtype=np.uint8)

    tile_size = TILES[0].shape[0]

    GameApp_PyGame(
        RubiksCross(
            CroixPharmaGraphics(
                TILES,
                colors,
                GraphicsMoveFunctionsMap(tile_size),
                animation_max_length=10),
            PyGameMixer(),
            difficulty=args.difficulty
        )).run()

    # GameApp_cv2(
    #     RubiksCross(
    #         CroixPharmaGraphics(TILES, colors, animation_max_length=10),
    #         SilentMixer(),
    #         difficulty=difficulty
    #     )).run()


if __name__ == '__main__':
    main()
