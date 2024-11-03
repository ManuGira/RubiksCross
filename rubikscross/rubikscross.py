import enum
import sys
import abc
import cv2
import numpy as np
import pygame
import numpy.typing as npt
import time
import functools

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
        frame_count = self.animation_max_length // 1
        res = []
        image0 = self.generate_image(board)
        factors = np.linspace(0, 1, frame_count + 1)[1:]
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

        center_xy = (w / 2 - 0.5, h / 2 - 0.5)
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

    def __init__(self, rcgraphics: RubiksCrossGraphicsInterface, difficulty: int = 2):
        self.rcgraphics: RubiksCrossGraphicsInterface = rcgraphics
        self.difficulty = difficulty

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
        self.init_board = np.array([
            [0, 1, 0],
            [2, 3, 4],
            [0, 5, 0],
        ], dtype=np.uint8).repeat(difficulty, axis=0).repeat(difficulty, axis=1)

        self.board: npt.NDArray
        self.reset()

    def reset(self):
        self.board = self.init_board.copy()
        self.rcgraphics.initialize_frame0(self.board)

    def on_action(self, action: 'RubiksCross.Action'):
        if action == RubiksCross.Action.SCRAMBLE:
            ind = np.random.randint(0, 4, 1)[0]
            for rn in np.random.randint(1, 4, 10 * self.difficulty ** 2):
                ind = (ind + 2 + rn) % 4  # avoid to take the opposit of previous move. (e.g. We don't want LEFT if it was RIGHT)
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
        return np.sum(abs(self.board - self.init_board).flatten()) == 0


class GameAppInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def set_image_u8(self, img_u8: npt.NDArray):
        """
        Sets the image to be displayed.

        The `image` argument should be a 2D numpy array of uint8 representing the pixels.
        Values range from 0 (off) to 255 (brightest).

        For float images, see function: set_image
        """
        pass

    @abc.abstractmethod
    def run(self):
        pass


class CV2(enum.IntEnum):
    """
    Keyboard mapping
    """
    K_ENTER = 13
    K_SPACE = 32
    K_a = 97
    K_d = 100
    K_e = 101
    K_q = 113
    K_s = 115
    K_w = 119
    K_LEFT = 2424832
    K_UP = 2490368
    K_RIGHT = 2555904
    K_DOWN = 2621440


class GameApp_cv2(GameAppInterface):
    SIZE = 720
    FPS = 60

    def __init__(self, rubikscross: RubiksCross):
        self.rubikscross = rubikscross
        self.time = 0

    def set_image_u8(self, img_u8: npt.NDArray):
        img_u8 = cv2.resize(img_u8, dsize=(GameApp_cv2.SIZE, GameApp_cv2.SIZE), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Rubik's Cross", img_u8)

    def run(self):
        inputs_action_map = {
            CV2.K_LEFT: RubiksCross.Action.LEFT,
            CV2.K_a: RubiksCross.Action.LEFT,
            CV2.K_RIGHT: RubiksCross.Action.RIGHT,
            CV2.K_d: RubiksCross.Action.RIGHT,
            CV2.K_UP: RubiksCross.Action.UP,
            CV2.K_w: RubiksCross.Action.UP,
            CV2.K_DOWN: RubiksCross.Action.DOWN,
            CV2.K_s: RubiksCross.Action.DOWN,
            CV2.K_e: RubiksCross.Action.ROT_RIGHT,
            CV2.K_q: RubiksCross.Action.ROT_LEFT,
            CV2.K_SPACE: RubiksCross.Action.SCRAMBLE,
        }

        next_frame_time = time.time()
        event_key_list = []
        frame_duration = 1 / GameApp_cv2.FPS
        while True:  # self.win.is_alive:
            now = time.time()
            sleep_time_ms = int((next_frame_time - now) * 1000)
            event_key = cv2.waitKeyEx(max(1, sleep_time_ms))

            if event_key != -1:
                event_key_list.append(event_key)
                continue

            while len(event_key_list) > 0:
                event_key = event_key_list.pop(0)
                if event_key in inputs_action_map.keys():
                    self.rubikscross.on_action(inputs_action_map[event_key])

            image = self.rubikscross.rcgraphics.get_next_frame()
            self.set_image_u8(image)
            next_frame_time += frame_duration
            next_frame_time = max(time.time(), next_frame_time)


class GameApp_PyGame(GameAppInterface):
    SIZE = 720
    FPS = 60

    def __init__(self, rubikscross: RubiksCross):
        self.rubikscross = rubikscross

        pygame.init()
        self.screen = pygame.display.set_mode(
            [GameApp_PyGame.SIZE, GameApp_PyGame.SIZE]
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.dot_mask: npt.NDArray

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def make_dot_mask_u16(grid_size):
        dot_size = GameApp_PyGame.SIZE // grid_size
        rad = int(round(dot_size/2))

        # working on bigger circle to get a nice blend
        dot = np.zeros((dot_size*10, dot_size*10), dtype=np.uint8)
        dot = cv2.circle(dot, (rad*10, rad*10), rad*10, (255,), thickness=-1)
        dot = cv2.resize(dot, dsize=(dot_size, dot_size), interpolation=cv2.INTER_AREA)

        dot_mask = cv2.repeat(dot.astype(np.uint16), grid_size, grid_size)

        if dot_mask.shape[0] != GameApp_PyGame.SIZE:
            dot_mask = cv2.resize(dot_mask, (GameApp_PyGame.SIZE, GameApp_PyGame.SIZE), interpolation=cv2.INTER_LINEAR)

        dot_mask.shape += (1,)
        return dot_mask

    def set_image_u8(self, img_u8: npt.NDArray):
        if img_u8.dtype != np.uint8:
            raise ValueError("Image type must be uint8")

        self.screen.fill((0, 0, 0))

        dot_mask_u16 = GameApp_PyGame.make_dot_mask_u16(img_u8.shape[0])
        img_u8 = cv2.transpose(img_u8)
        img_u8 = cv2.resize(img_u8, dsize=(GameApp_PyGame.SIZE, GameApp_PyGame.SIZE), interpolation=cv2.INTER_NEAREST)

        img_u8 = ((dot_mask_u16 * img_u8.astype(np.uint16))//255).astype(np.uint8)

        surface = pygame.surfarray.make_surface(img_u8)
        self.screen.blit(surface, (0, 0))

    def run(self):
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
                        self.rubikscross.on_action(inputs_action_map[event.key])

            image = self.rubikscross.rcgraphics.get_next_frame()
            self.set_image_u8(image)

            current_fps = self.clock.get_fps()
            fps_img = self.font.render(f"FPS: {current_fps:.1f}", True, (0, 100, 0))
            self.screen.blit(fps_img, (0, 0))
            pygame.display.flip()
            self.frame_timing = self.clock.tick(GameApp_PyGame.FPS)


def main_game(difficulty: int = 2):
    assert difficulty < 11
    colors = np.array([
        [0, 0, 0],
        [249, 220, 92],
        [245, 100, 118],
        [98, 113, 231],
        [61, 204, 202],
        [209, 217, 210],
    ], dtype=np.uint8)

    tiles = []
    for tile, color in zip(TILES, colors):
        tile = tile.reshape((*tile.shape, 1))
        color = color.reshape((1, 1, 3))
        tiles.append((1 - tile) * color)

    cpgraphics = CroixPharamGraphics(tiles, animation_max_length=10)
    rubikscross = RubiksCross(cpgraphics, difficulty=difficulty)

    GameApp_PyGame(rubikscross).run()
    # GameApp_cv2(rubikscross).run()


if __name__ == '__main__':
    main_game(3)
