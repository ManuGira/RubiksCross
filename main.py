from rubikscross.game_app import GameApp_PyGame
from rubikscross.graphic_painters import RollFuncMap
from rubikscross.graphics import Graphics
from rubikscross.mixer import PyGameMixer
from rubikscross.rubikscross import RubiksCross
from rubikscross.tiles import TILES


def main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="Rubik's Cross",
        description="A kind of Rubik's Cube but in 2D and in cross shape. Controls: Press SPACE to scramble the puzzle. Use directional arrows (or WASD) to solve it. Press Q and E for 90° rotations",
    )
    parser.add_argument("-d", "--difficulty", type=int, default=2)
    import sys
    args=parser.parse_args(sys.argv[1:])

    assert args.difficulty < 11

    GameApp_PyGame(
        RubiksCross(
            Graphics(
                TILES,
                RollFuncMap(TILES[0].shape[0], GameApp_PyGame.SIZE),
                # FadeFuncMap(tile_size, GameApp_PyGame.SIZE),
                # CoolFuncMap(tile_size, GameApp_PyGame.SIZE, args.difficulty),
                animation_max_length=100),
            PyGameMixer(),
            difficulty=args.difficulty
        )).run()

    # GameApp_cv2(
    #     RubiksCross(
    #         Graphics(
    #             TILES,
    #             RollFuncMap(TILES[0].shape[0], GameApp_PyGame.SIZE),
    #             animation_max_length=100),
    #         SilentMixer(),
    #         difficulty=args.difficulty
    #     )).run()


if __name__ == '__main__':
    main()
