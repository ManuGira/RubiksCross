import enum
import sys
import abc
import cv2
import numpy as np
import pygame
import numpy.typing as npt
import time
import functools
import controller_pro

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


class RubiksCrossMixerInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def play_sound(self, action: "RubiksCross.Action"):
        pass


class SilentMixer(RubiksCrossMixerInterface):
    def play_sound(self, action):
        pass


class PyGameMixer(RubiksCrossMixerInterface):
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        self.sounds = {
            RubiksCross.Action.UP: pygame.mixer.Sound(f"assets/Y1.mp3"),
            RubiksCross.Action.DOWN: pygame.mixer.Sound(f"assets/Y0.mp3"),
            RubiksCross.Action.LEFT: pygame.mixer.Sound(f"assets/X0.mp3"),
            RubiksCross.Action.RIGHT: pygame.mixer.Sound(f"assets/X1.mp3"),
            RubiksCross.Action.ROT_LEFT: pygame.mixer.Sound(f"assets/Z0.mp3"),
            RubiksCross.Action.ROT_RIGHT: pygame.mixer.Sound(f"assets/Z1.mp3"),
        }

    def play_sound(self, action):
        if action in self.sounds.keys():
            pygame.mixer.Sound.play(self.sounds[action])


class RubiksCrossGraphicsInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def initialize_frame(self, board: npt.NDArray):
        pass

    @abc.abstractmethod
    def initialize_hint_frame(self, board: npt.NDArray):
        pass

    @abc.abstractmethod
    def generate_frame(self, board: npt.NDArray) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def update_animation(self, move_func, board: npt.NDArray, frame_count: int | None = None):
        pass

    @abc.abstractmethod
    def get_next_frame(self, height: int | None = None) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def get_hint_frame(self) -> npt.NDArray:
        pass

    @abc.abstractmethod
    def get_tile_size(self) -> int:
        pass


class CroixPharamGraphics(RubiksCrossGraphicsInterface):
    def __init__(self, tiles: list[npt.NDArray], colors: npt.NDArray, animation_max_length: int):
        self.colors = colors.copy()

        # colorize tiles
        self.tiles = []
        for tile, color in zip(tiles, colors):
            tile = tile.reshape((*tile.shape, 1))
            color = color.reshape((1, 1, 3))
            self.tiles.append((1 - tile) * color)

        self.tile_size = self.tiles[0].shape[0]
        self.animation_max_length: int = animation_max_length
        self.frame_config_list: list[tuple[Callable, npt.NDArray]] = []
        self.frame_continuous_index: int = 0
        self.frame: npt.NDArray
        self.hint_frame: npt.NDArray

    def initialize_frame(self, board):
        self.frame = self.generate_frame(board)

    def initialize_hint_frame(self, initial_board):
        self.hint_frame = self.generate_frame(initial_board.copy())

    @staticmethod
    def insert_tile(image, tile, coord_ij):
        th, tw = tile.shape[:2]
        i, j = coord_ij
        image[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = tile

    def generate_frame(self, board: npt.NDArray) -> npt.NDArray:
        """
        This will create an image made of tiles, placed according to the board.

        Each value in the board is used as an index. This index is used to select a tile in the tile list

        Parameters
        ----------
        board: (m0, n0) array

        Returns
        -------
        An array of dimension (m0*m1, n0*n1) where (m1, n1) is the dimension of the tiles
        """
        bsize = board.shape[0]
        tsize = self.tiles[0].shape[0]
        frame_size = bsize * tsize

        res = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

        for i in range(bsize):
            for j in range(bsize):
                ind = int(board[i, j])
                CroixPharamGraphics.insert_tile(res, self.tiles[ind], (i, j))
        return res

    def update_animation(self, move_func: callable, board: npt.NDArray, frame_count: int | None = None):
        self.frame_config_list.append((move_func, board.copy()))

    def get_next_frame(self, height: int | None = None) -> npt.NDArray:
        fci_max = len(self.frame_config_list)
        remaining_length = fci_max - self.frame_continuous_index

        # update index and avoid out of bounds
        min_step = 1 / 20
        step = max(remaining_length / 5, min_step)
        self.frame_continuous_index = min(self.frame_continuous_index + step, fci_max)

        #  frame_continuous_index is N+t, where N is an integer and t a float in interval [0, 1[
        N = int(self.frame_continuous_index)
        t = self.frame_continuous_index % 1.0

        if N == len(self.frame_config_list) and N > 0:
            N -= 1
            t += 1

        # drop the N first configs
        self.frame_config_list = self.frame_config_list[N:]
        need_new_frame = self.frame_continuous_index > min_step / 2 and len(self.frame_config_list) > 0
        if need_new_frame:
            # generate the frame given its config
            move_func, board = self.frame_config_list[0]
            tiled_board = self.generate_frame(board)
            self.frame = move_func(tiled_board, t)

        res = self.frame
        if height is not None:
            # resize it to given height
            h = res.shape[0]
            f = height / h
            res = cv2.resize(res, dsize=None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)

        self.frame_continuous_index = t
        return res

    def get_hint_frame(self) -> npt.NDArray:
        return self.hint_frame

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
        SAVE1 = enum.auto()
        SAVE2 = enum.auto()
        SAVE3 = enum.auto()
        LOAD1 = enum.auto()
        LOAD2 = enum.auto()
        LOAD3 = enum.auto()

    class State(enum.Enum):
        FREE = enum.auto()
        SCRAMBLING = enum.auto()
        READY = enum.auto()
        RACING = enum.auto()

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

    def __init__(self, rcgraphics: RubiksCrossGraphicsInterface, rcmixer: RubiksCrossMixerInterface, difficulty: int = 2):
        self.rcgraphics: RubiksCrossGraphicsInterface = rcgraphics
        self.rcmixer: RubiksCrossMixerInterface = rcmixer
        self.difficulty = difficulty
        self.state = RubiksCross.State.FREE

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
        self.save_actions = [
            RubiksCross.Action.SAVE1,
            RubiksCross.Action.SAVE2,
            RubiksCross.Action.SAVE3,
        ]
        self.load_actions = [
            RubiksCross.Action.LOAD1,
            RubiksCross.Action.LOAD2,
            RubiksCross.Action.LOAD3,
        ]
        self.memory_actions = self.save_actions + self.load_actions

        self.init_board = np.array([
            [0, 1, 0],
            [2, 3, 4],
            [0, 5, 0],
        ], dtype=np.uint8).repeat(difficulty, axis=0).repeat(difficulty, axis=1)

        self.board: npt.NDArray
        self.chrono: float
        self.time0: float
        self.move_count: float
        self.reset()

        self.saved_boards = [self.init_board.copy() for _ in range(len(self.save_actions))]

    def reset(self):
        self.chrono = 0
        self.move_count = 0
        self.state = RubiksCross.State.FREE
        self.board = self.init_board.copy()
        self.rcgraphics.initialize_frame(self.board)
        self.rcgraphics.initialize_hint_frame(self.init_board)

    def save_board(self, slot_id):
        self.state = RubiksCross.State.FREE
        self.saved_boards[slot_id] = self.board.copy()

    def load_board(self, slot_id):
        self.state = RubiksCross.State.FREE
        self.board = self.saved_boards[slot_id].copy()
        self.rcgraphics.generate_frame(self.board)

    def update_chrono(self):
        if self.state == RubiksCross.State.RACING:
            self.chrono = time.time() - self.time0
        return self.chrono

    def on_action(self, action: 'RubiksCross.Action', mute_sound: bool = False, frame_count: int | None = None):
        if action == RubiksCross.Action.SCRAMBLE:
            self.reset()
            self.state = RubiksCross.State.SCRAMBLING
            ind = np.random.randint(0, 4, 1)[0]
            for rn in np.random.randint(1, 4, 10 * self.difficulty ** 2):
                ind = (ind + 2 + rn) % 4  # avoid to take the opposite of previous move. (e.g. We don't want LEFT if it was RIGHT)
                action = [RubiksCross.Action.LEFT, RubiksCross.Action.UP, RubiksCross.Action.RIGHT, RubiksCross.Action.DOWN][ind]
                self.on_action(action, mute_sound=True, frame_count=1)
            self.state = RubiksCross.State.READY
        elif action in self.save_actions:
            slot_ind = self.save_actions.index(action)
            self.save_board(slot_ind)
        elif action in self.load_actions:
            slot_ind = self.load_actions.index(action)
            self.load_board(slot_ind)
        else:
            move_func = self.action_func_map[action]
            tile_size = self.rcgraphics.get_tile_size()

            anim_move_func = move_func
            if action in self.roll_actions:
                anim_move_func = lambda b, f: move_func(board=b, shift=tile_size, factor=f)

            if not mute_sound:
                self.rcmixer.play_sound(action)

            self.rcgraphics.update_animation(anim_move_func, self.board, frame_count)
            self.board = move_func(self.board)

            match self.state:
                case RubiksCross.State.READY:
                    self.time0 = time.time()
                    self.state = RubiksCross.State.RACING
                    self.move_count = 1
                case RubiksCross.State.RACING:
                    self.move_count += 1
                    if self.is_solved():
                        self.state = RubiksCross.State.FREE

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

            image = self.rubikscross.rcgraphics.get_next_frame(GameApp_cv2.SIZE)
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
    def make_dot_mask(grid_size):
        dot_size = GameApp_PyGame.SIZE // grid_size
        magn = 10
        center = int(round(magn * dot_size / 2))
        rad = int(round(magn * dot_size / 2))

        # working on bigger circle to get a nice blend
        dot = np.zeros((dot_size * magn, dot_size * magn), dtype=np.uint8)
        dot = cv2.circle(dot, (center, center), rad, (255,), thickness=-1)
        dot = cv2.resize(dot, dsize=(dot_size, dot_size), interpolation=cv2.INTER_AREA)

        dot_mask = cv2.repeat(dot, grid_size, grid_size)

        if dot_mask.shape[0] != GameApp_PyGame.SIZE:
            dot_mask = cv2.resize(dot_mask, (GameApp_PyGame.SIZE, GameApp_PyGame.SIZE), interpolation=cv2.INTER_LINEAR)

        return dot_mask

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def make_square_mask(grid_size):
        square_size = GameApp_PyGame.SIZE // grid_size
        rad_ratio = 0.8
        topleft = int(round((1 - rad_ratio) * square_size / 2))
        botright = int(round((1 + rad_ratio) * square_size / 2))

        square = np.zeros((square_size, square_size), dtype=np.uint8)
        square[topleft:botright, topleft:botright] = 255

        square_mask = cv2.repeat(square, grid_size, grid_size)

        if square_mask.shape[0] != GameApp_PyGame.SIZE:
            square_mask = cv2.resize(square_mask, (GameApp_PyGame.SIZE, GameApp_PyGame.SIZE), interpolation=cv2.INTER_NEAREST)

        return square_mask

    def set_image_u8(self, img_u8: npt.NDArray):
        if img_u8.dtype != np.uint8:
            raise ValueError("Image type must be uint8")

        # dot_mask = GameApp_PyGame.make_dot_mask(img_u8.shape[0]).copy()
        dot_mask = GameApp_PyGame.make_square_mask(img_u8.shape[0]).copy()
        img_u8 = cv2.transpose(img_u8)
        img_u8 = cv2.resize(img_u8, dsize=(GameApp_PyGame.SIZE, GameApp_PyGame.SIZE), interpolation=cv2.INTER_NEAREST)

        dot_mask *= (cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY) != 0).astype(np.uint8)

        surface = pygame.surfarray.make_surface(img_u8)
        surface = surface.convert_alpha()  # Enable per-pixel alpha
        pygame.surfarray.pixels_alpha(surface)[:, :] = dot_mask
        self.screen.blit(surface, (0, 0))

    def run(self):
        key_action_map = {
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
            pygame.K_1: RubiksCross.Action.SAVE1,
            pygame.K_2: RubiksCross.Action.SAVE2,
            pygame.K_3: RubiksCross.Action.SAVE3,
            pygame.K_F1: RubiksCross.Action.LOAD1,
            pygame.K_F2: RubiksCross.Action.LOAD2,
            pygame.K_F3: RubiksCross.Action.LOAD3,
        }

        key_action_map.update({
            controller_pro.SwitchProController.J_LEFT: RubiksCross.Action.LEFT,
            controller_pro.SwitchProController.J_L: RubiksCross.Action.LEFT,
            controller_pro.SwitchProController.J_RIGHT: RubiksCross.Action.RIGHT,
            controller_pro.SwitchProController.J_R: RubiksCross.Action.RIGHT,
            controller_pro.SwitchProController.J_UP: RubiksCross.Action.UP,
            controller_pro.SwitchProController.J_ZL: RubiksCross.Action.UP,
            controller_pro.SwitchProController.J_DOWN: RubiksCross.Action.DOWN,
            controller_pro.SwitchProController.J_ZR: RubiksCross.Action.DOWN,
            controller_pro.SwitchProController.J_A: RubiksCross.Action.ROT_RIGHT,
            controller_pro.SwitchProController.J_B: RubiksCross.Action.ROT_LEFT,
            controller_pro.SwitchProController.J_PLUS: RubiksCross.Action.SCRAMBLE,
        })

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type in [pygame.KEYDOWN]:
                    if event.key in key_action_map.keys():
                        self.rubikscross.on_action(key_action_map[event.key])
                elif event.type in [pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN]:
                    controller_event = controller_pro.controller_pro_event(event)
                    if controller_event in key_action_map.keys():
                        self.rubikscross.on_action(key_action_map[controller_event])


            self.screen.fill((0, 0, 0))

            hint_frame = self.rubikscross.rcgraphics.get_hint_frame()
            hint_frame = cv2.transpose(hint_frame)
            hint_frame = cv2.resize(hint_frame, dsize=(GameApp_PyGame.SIZE // 5, GameApp_PyGame.SIZE // 5), interpolation=cv2.INTER_NEAREST)
            background_surface = pygame.surfarray.make_surface(hint_frame)
            self.screen.blit(background_surface, (
                GameApp_PyGame.SIZE * 11 // 15,
                GameApp_PyGame.SIZE // 15,
            ))

            image = self.rubikscross.rcgraphics.get_next_frame()
            self.set_image_u8(image)

            current_fps = self.clock.get_fps()
            fps_img = self.font.render(f"FPS: {current_fps:.1f}", True, (0, 100, 0))
            self.screen.blit(fps_img, (5, 5))

            chrono_img = self.font.render(f"Timer: {self.rubikscross.update_chrono(): 7.2f}", True, (0, 100, 0))
            self.screen.blit(chrono_img, (5, 25))

            chrono_img = self.font.render(f"Moves: {self.rubikscross.move_count:4}", True, (0, 100, 0))
            self.screen.blit(chrono_img, (5, 45))

            pygame.display.flip()
            self.frame_timing = self.clock.tick(GameApp_PyGame.FPS)


def main_game(difficulty: int = 2):
    assert difficulty < 11
    colors = np.array([
        [0, 0, 0],  # black
        [61, 204, 202],  # cyan
        [245, 100, 118],  # red
        [209, 217, 210],  # light gray
        [249, 220, 92],  # yellow
        [98, 113, 231],  # dark blue
    ], dtype=np.uint8)

    GameApp_PyGame(
        RubiksCross(
            CroixPharamGraphics(TILES, colors, animation_max_length=10),
            PyGameMixer(),
            difficulty=difficulty
        )).run()

    # GameApp_cv2(
    #     RubiksCross(
    #         CroixPharamGraphics(TILES, colors, animation_max_length=10),
    #         SilentMixer(),
    #         difficulty=difficulty
    #     )).run()


if __name__ == '__main__':
    main_game(2)
