import enum
import enum
import functools

import cv2
import numpy as np
import numpy.typing as npt
import pygame

import controller_pro
from actions import Action
from interfaces import GameAppInterface
from rubikscross import RubiksCross


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
            CV2.K_LEFT: Action.LEFT,
            CV2.K_a: Action.LEFT,
            CV2.K_RIGHT: Action.RIGHT,
            CV2.K_d: Action.RIGHT,
            CV2.K_UP: Action.UP,
            CV2.K_w: Action.UP,
            CV2.K_DOWN: Action.DOWN,
            CV2.K_s: Action.DOWN,
            CV2.K_e: Action.ROT_RIGHT,
            CV2.K_q: Action.ROT_LEFT,
            CV2.K_SPACE: Action.SCRAMBLE,
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
            pygame.K_LEFT: Action.LEFT,
            pygame.K_a: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_d: Action.RIGHT,
            pygame.K_UP: Action.UP,
            pygame.K_w: Action.UP,
            pygame.K_DOWN: Action.DOWN,
            pygame.K_s: Action.DOWN,
            pygame.K_e: Action.ROT_RIGHT,
            pygame.K_q: Action.ROT_LEFT,
            pygame.K_SPACE: Action.SCRAMBLE,
            pygame.K_1: Action.SAVE1,
            pygame.K_2: Action.SAVE2,
            pygame.K_3: Action.SAVE3,
            pygame.K_F1: Action.LOAD1,
            pygame.K_F2: Action.LOAD2,
            pygame.K_F3: Action.LOAD3,
        }

        key_action_map.update({
            controller_pro.SwitchProController.J_LEFT: Action.LEFT,
            controller_pro.SwitchProController.J_L: Action.LEFT,
            controller_pro.SwitchProController.J_RIGHT: Action.RIGHT,
            controller_pro.SwitchProController.J_R: Action.RIGHT,
            controller_pro.SwitchProController.J_UP: Action.UP,
            controller_pro.SwitchProController.J_ZL: Action.UP,
            controller_pro.SwitchProController.J_DOWN: Action.DOWN,
            controller_pro.SwitchProController.J_ZR: Action.DOWN,
            controller_pro.SwitchProController.J_A: Action.ROT_RIGHT,
            controller_pro.SwitchProController.J_B: Action.ROT_LEFT,
            controller_pro.SwitchProController.J_PLUS: Action.SCRAMBLE,
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
