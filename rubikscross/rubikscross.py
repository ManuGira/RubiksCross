import enum
import time

import cv2
import numpy as np
import numpy.typing as npt

from . import actions
from .actions import Action
from .interfaces import MixerInterface
from .graphics import Graphics


class RubiksCross:
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

    def __init__(self, rcgraphics: Graphics, rcmixer: MixerInterface, difficulty: int = 2):
        self.rcgraphics: GraphicsInterface = rcgraphics
        self.rcmixer: MixerInterface = rcmixer
        self.difficulty = difficulty
        self.grid_size = 3 * difficulty
        self.state = RubiksCross.State.FREE
        self.action_func_map = {
            Action.RIGHT: RubiksCross.cross_roll_right,
            Action.LEFT: RubiksCross.cross_roll_left,
            Action.UP: RubiksCross.cross_roll_up,
            Action.DOWN: RubiksCross.cross_roll_down,
            Action.ROT_RIGHT: RubiksCross.cross_rot90_right,
            Action.ROT_LEFT: RubiksCross.cross_rot90_left,
        }

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

        self.saved_boards = [self.init_board.copy() for _ in range(len(actions.save_actions))]

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
        self.rcgraphics.reset_frame_config(self.board)

    def update_chrono(self):
        if self.state == RubiksCross.State.RACING:
            self.chrono = time.time() - self.time0
        return self.chrono

    def on_action(self, action: Action, mute_sound: bool = False, frame_count: int | None = None):
        if action == Action.SCRAMBLE:
            self.reset()
            self.state = RubiksCross.State.SCRAMBLING
            ind = np.random.randint(0, 4, 1)[0]
            for rn in np.random.randint(1, 4, 10 * self.difficulty ** 2):
                ind = (ind + 2 + rn) % 4  # avoid to take the opposite of previous move. (e.g. We don't want LEFT if it was RIGHT)
                action = [Action.LEFT, Action.UP, Action.RIGHT, Action.DOWN][ind]
                self.on_action(action, mute_sound=True, frame_count=1)
            self.state = RubiksCross.State.READY
        elif action in actions.save_actions:
            slot_ind = actions.save_actions.index(action)
            self.save_board(slot_ind)
        elif action in actions.load_actions:
            slot_ind = actions.load_actions.index(action)
            self.load_board(slot_ind)
        else:
            move_func = self.action_func_map[action]

            if not mute_sound:
                self.rcmixer.play_sound(action)

            self.rcgraphics.update_animation(action, self.board, frame_count)
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
