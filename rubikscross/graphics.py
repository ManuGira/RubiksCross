import cv2
import numpy as np
import numpy.typing as npt

from .actions import Action
from .tiles import t_tilesbook
from .interfaces import GraphicPainterInterface


class Graphics:
    def __init__(self, tiles: t_tilesbook, move_func_map: GraphicPainterInterface, animation_max_length: int):
        # colorize tiles
        self.tiles: t_tilesbook = tiles
        self.move_func_map = move_func_map
        self.animation_max_length: int = animation_max_length

        self.tile_size: int = self.tiles[0].shape[0]
        self.frame_config_list: list[tuple[Callable, npt.NDArray]] = []
        self.frame_continuous_index: int = 0
        self.frame: npt.NDArray
        self.alpha: npt.NDArray
        self.hint_frame: npt.NDArray

    def initialize_frame(self, board):
        self.frame, self.alpha = self.move_func_map.compute(Action.RIGHT, board, self.tiles, factor=0)

    def initialize_hint_frame(self, initial_board):
        self.hint_frame, self.alpha = self.move_func_map.compute(Action.RIGHT, initial_board, self.tiles, factor=0)

    def update_animation(self, action: Action, board: npt.NDArray, frame_count: int | None = None):
        self.frame_config_list.append((action, board.copy()))

    def get_next_frame(self, height: int | None = None) -> tuple[npt.NDArray, npt.NDArray]:
        fci_max = len(self.frame_config_list)
        remaining_length = fci_max - self.frame_continuous_index

        # update index and avoid out of bounds
        N = 50
        min_step = 1 / N
        step = max(remaining_length * 4 / N, min_step)
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
            action, board = self.frame_config_list[0]
            self.frame, self.alpha = self.move_func_map.compute(action, board=board, tiles=self.tiles, factor=t)

        self.frame_continuous_index = t
        return self.frame, self.alpha

    def get_hint_frame(self) -> npt.NDArray:
        return self.hint_frame

    def get_tile_size(self) -> int:
        return self.tile_size
