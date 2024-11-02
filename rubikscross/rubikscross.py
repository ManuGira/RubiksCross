import enum
import sys
import abc
import cv2
import numpy as np
import pygame
import numpy.typing as npt

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


class RubiksCrossGraphicsInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def initialize_frame0(self, board: npt.NDArray):
        pass

    @abc.abstractmethod
    def update_animation(self, move_func, board: npt.NDArray):
        pass

    @abc.abstractmethod
    def get_next_frame(self) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def get_tile_size(self) -> int:
        pass


class CroixPharamGraphics(RubiksCrossGraphicsInterface):
    def __init__(self, tiles: list[npt.NDArray], animation_max_length: int):
        self.tiles: list[npt.NDArray] = tiles
        self.tile_size = self.tiles[0].shape[0]
        self.is_rgb: bool = len(tiles[0].shape) == 3
        self.animation_max_length: int = animation_max_length
        self.animation: list[npt.NDArray] = []
        self.frame: npt.NDArray

    def initialize_frame0(self, board):
        self.frame = self.generate_image(board)

    @staticmethod
    def insert_tile(image, tile, coord_ij):
        th, tw = tile.shape[:2]
        i, j = coord_ij
        image[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tile

    def generate_image(self, board: npt.NDArray):
        bh, bw = board.shape
        th, tw = self.tiles[0].shape[:2]
        frame_height = th * bh
        frame_width = tw * bw

        if self.is_rgb:
            res = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        else:
            res = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for i in range(bh):
            for j in range(bw):
                ind = int(board[i, j])
                CroixPharamGraphics.insert_tile(res, self.tiles[ind], (i, j))
        return res

    def drop_frames(self, animation: list[npt.NDArray], target_length):
        current_length = len(animation)
        drop_count = current_length - target_length
        for i in range(drop_count)[::-1]:
            ind = (i * current_length) // drop_count
            animation.pop(ind)

    def generate_animation(self, move_func, board: npt.NDArray):
        frame_count = self.animation_max_length // 2
        res = []
        image0 = self.generate_image(board)
        factors = np.linspace(0, 1, frame_count+1)[1:]
        for factor in factors:
            frame = move_func(image0, factor)
            res.append(frame)
        return res

    def update_animation(self, move_func, board: npt.NDArray):
        new_frames = self.generate_animation(move_func, board)
        self.animation += new_frames
        self.drop_frames(self.animation, self.animation_max_length)

    def get_next_frame(self) -> npt.NDArray:
        if len(self.animation) > 0:
            self.frame = self.animation.pop(0)
        return self.frame

    def get_tile_size(self) -> int:
        return self.tile_size


class RubiksCross:
    class Action(enum.Enum):
        LEFT = enum.auto()
        RIGHT = enum.auto()
        UP = enum.auto()
        DOWN = enum.auto()
        ROT_LEFT = enum.auto()
        ROT_RIGHT = enum.auto()
        SCRAMBLE = enum.auto()

    @staticmethod
    def cross_rot90_right(board, factor: float = 1.0):
        h, w = board.shape[:2]
        assert h % 3 == 0
        assert h == w

        center_xy = (w/2-0.5, h/2-0.5)
        mat = cv2.getRotationMatrix2D(center=center_xy, angle=-90 * factor, scale=1)
        board = cv2.warpAffine(board, mat, (w, h))
        return board

    @staticmethod
    def cross_rot90_left(board, factor: float = 1.0):
        return RubiksCross.cross_rot90_right(board, -factor)

    @staticmethod
    def cross_roll_right(board: npt.NDArray, shift: int = 1, factor: float = 1.0):
        h, w = board.shape[:2]
        assert h % 3 == 0
        assert h == w

        shift = int(round(shift * factor))
        n = h // 3
        board = board.copy()
        board[:n, n:2 * n] = np.roll(board[:n, n:2 * n], shift=shift, axis=1)
        board[n:2 * n, :] = np.roll(board[n:2 * n, :], shift=shift, axis=1)
        board[2 * n:, n:2 * n] = np.roll(board[2 * n:, n:2 * n], shift=shift, axis=1)
        return board

    @staticmethod
    def cross_roll_left(board: npt.NDArray, shift: int = 1, factor: float = 1.0):
        return RubiksCross.cross_roll_right(board, -shift, factor)

    @staticmethod
    def cross_roll_up(board: npt.NDArray, shift: int = 1, factor: float = 1.0):
        return RubiksCross.cross_roll_left(board.swapaxes(0, 1), shift, factor).swapaxes(0, 1)

    @staticmethod
    def cross_roll_down(board: npt.NDArray, shift: int = 1, factor: float = 1.0):
        return RubiksCross.cross_roll_right(board.swapaxes(0, 1), shift, factor).swapaxes(0, 1)

    def __init__(self, rcgraphics: RubiksCrossGraphicsInterface):
        self.rcgraphics: RubiksCrossGraphicsInterface = rcgraphics

        self.action_func_map = {
            RubiksCross.Action.RIGHT: RubiksCross.cross_roll_right,
            RubiksCross.Action.LEFT: RubiksCross.cross_roll_left,
            RubiksCross.Action.UP: RubiksCross.cross_roll_up,
            RubiksCross.Action.DOWN: RubiksCross.cross_roll_down,
            RubiksCross.Action.ROT_RIGHT: RubiksCross.cross_rot90_right,
            RubiksCross.Action.ROT_LEFT: RubiksCross.cross_rot90_left,
        }
        self.roll_actions = [
            RubiksCross.Action.RIGHT,
            RubiksCross.Action.LEFT,
            RubiksCross.Action.UP,
            RubiksCross.Action.DOWN,
        ]
        self.rot_actions = [
            RubiksCross.Action.ROT_RIGHT,
            RubiksCross.Action.ROT_LEFT,
        ]

        self.init_2x2x5 = np.array([
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [2, 2, 3, 3, 4, 4],
            [2, 2, 3, 3, 4, 4],
            [0, 0, 5, 5, 0, 0],
            [0, 0, 5, 5, 0, 0],
        ], dtype=np.uint8)

        self.board: npt.NDArray
        self.reset()

    def reset(self):
        self.board = self.init_2x2x5.copy()
        self.rcgraphics.initialize_frame0(self.board)

    def on_action(self, action: 'RubiksCross.Action'):
        if action == RubiksCross.Action.SCRAMBLE:
            ind = np.random.randint(0, 4, 1)[0]
            for rn in np.random.randint(1, 4, 10):
                ind = (ind+2+rn) % 4  # avoid to take the opposit of previous move. (e.g. We don't want LEFT if it was RIGHT)
                action = [RubiksCross.Action.LEFT, RubiksCross.Action.UP, RubiksCross.Action.RIGHT, RubiksCross.Action.DOWN][ind]
                self.on_action(action)
        else:
            move_func = self.action_func_map[action]
            tile_size = self.rcgraphics.get_tile_size()

            anim_move_func = move_func
            if action in self.roll_actions:
                anim_move_func = lambda b, f: move_func(board=b, shift=tile_size, factor=f)

            self.rcgraphics.update_animation(anim_move_func, self.board)
            self.board = move_func(self.board)

    def is_solved(self):
        return np.sum(abs(self.board - self.init_2x2x5).flatten()) == 0


class GameScreen:
    PANEL_SIZE = 16  # Size of a single panel on the cross, in pixels
    SCREEN_SIZE = 3 * PANEL_SIZE  # Width and height of the cross, in pixels
    PIXEL_SIZE = 16  # Width of each square representing an LED
    PIXEL_RADIUS_RATIO = 1  # Relative diameter of each LED

    def __init__(self):
        self.local_screen = pygame.display.set_mode(
            [GameScreen.PIXEL_SIZE * GameScreen.SCREEN_SIZE, GameScreen.PIXEL_SIZE * GameScreen.SCREEN_SIZE]
        )
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.font = pygame.font.SysFont(None, 24)

    def set_image_u8(self, img_u8: npt.NDArray):
        """
        Sets the image to be displayed.

        The `image` argument should be a 2D numpy array of uint8 representing the pixels.
        Values range from 0 (off) to 255 (brightest).

        For float images, see function: set_image
        """
        if img_u8.shape[:2] != (GameScreen.SCREEN_SIZE, GameScreen.SCREEN_SIZE):
            raise ValueError(
                f"Invalid image size (expected {GameScreen.SCREEN_SIZE}x{GameScreen.SCREEN_SIZE}, got {img_u8.shape})"
            )

        if img_u8.dtype != np.uint8:
            raise ValueError("Image type must be uint8")

        self.local_screen.fill((0, 0, 0))

        for i in range(img_u8.shape[0]):
            for j in range(img_u8.shape[1]):
                pixel_rgb = img_u8[i, j]
                try:
                    if sum(abs(pixel_rgb).flatten()) == 0:
                        continue
                except:
                    print()
                center_xy = (GameScreen.PIXEL_SIZE * (j + 0.5), GameScreen.PIXEL_SIZE * (i + 0.5))
                pygame.draw.circle(
                    self.local_screen,
                    pixel_rgb,
                    center_xy,
                    GameScreen.PIXEL_SIZE * GameScreen.PIXEL_RADIUS_RATIO / 2,
                )

        current_fps = self.clock.get_fps()
        fps_img = self.font.render(f"FPS: {current_fps:.1f}", True, (0, 100, 0))
        self.local_screen.blit(fps_img, (0, 0))
        pygame.display.flip()
        self.frame_timing = self.clock.tick(self.fps)

    def set_image(self, image: list[list[float]]):
        """
        Sets the image to be displayed.

        The `image` argument should be an array of floats representing the pixels in (row, column) order.
        Values range from 0.0 (off) to 1.0 (brightest).
        Note that 4 sections of the image will be ignored as the screen is a cross.

        For uint8 images, see function: set_image_u8
        """
        arr = np.array(image, dtype=float)

        if arr.shape != (GameScreen.SCREEN_SIZE, GameScreen.SCREEN_SIZE):
            raise ValueError(
                f"Invalid image size (expected {GameScreen.SCREEN_SIZE}x{GameScreen.SCREEN_SIZE}, got {arr.shape})"
            )

        if arr.flatten().min() < 0.0 or 1.0 < arr.flatten().max():
            raise ValueError("Pixel values must be between 0.0 and 1.0")

        im8 = np.round(arr * 255).astype(np.uint8)
        self.set_image_u8(im8)


def run_rubikscross(screen, tiles):
    cpgraphics = CroixPharamGraphics(tiles, animation_max_length=15)
    rubikscross = RubiksCross(cpgraphics)

    inputs_action_map = {
        pygame.K_LEFT: RubiksCross.Action.LEFT,
        pygame.K_a: RubiksCross.Action.LEFT,
        pygame.K_RIGHT: RubiksCross.Action.RIGHT,
        pygame.K_d: RubiksCross.Action.RIGHT,
        pygame.K_UP: RubiksCross.Action.UP,
        pygame.K_w: RubiksCross.Action.UP,
        pygame.K_DOWN: RubiksCross.Action.DOWN,
        pygame.K_s: RubiksCross.Action.DOWN,
        pygame.K_e: RubiksCross.Action.ROT_RIGHT,
        pygame.K_q: RubiksCross.Action.ROT_LEFT,
        pygame.K_SPACE: RubiksCross.Action.SCRAMBLE,
    }

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key in inputs_action_map.keys():
                    rubikscross.on_action(inputs_action_map[event.key])

        image = rubikscross.rcgraphics.get_next_frame()
        screen.set_image_u8(image)


def main_game():
    pygame.init()
    screen = GameScreen()

    colors = np.array([
        [0, 0, 0],
        [87, 77, 104],
        [100, 143, 133],
        [249, 220, 92],
        [245, 100, 118],
        [209, 217, 210],
    ], dtype=np.uint8)

    tiles = []
    for tile, color in zip(TILES, colors):
        tile = tile.reshape((*tile.shape, 1))
        color = color.reshape((1, 1, 3))
        tiles.append((1 - tile) * color)

    run_rubikscross(screen, tiles)


if __name__ == '__main__':
    # main_croixpharmacie()
    main_game()
