import os

import pygame

from actions import Action
from interfaces import MixerInterface


class SilentMixer(MixerInterface):
    def play_sound(self, action):
        pass


class PyGameMixer(MixerInterface):
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]

        dirname = os.path.dirname(__file__)
        self.sounds = {
            Action.UP: pygame.mixer.Sound(os.path.join(dirname, "assets/Y1.mp3")),
            Action.DOWN: pygame.mixer.Sound(os.path.join(dirname, "assets/Y0.mp3")),
            Action.LEFT: pygame.mixer.Sound(os.path.join(dirname, "assets/X0.mp3")),
            Action.RIGHT: pygame.mixer.Sound(os.path.join(dirname, "assets/X1.mp3")),
            Action.ROT_LEFT: pygame.mixer.Sound(os.path.join(dirname, "assets/Z0.mp3")),
            Action.ROT_RIGHT: pygame.mixer.Sound(os.path.join(dirname, "assets/Z1.mp3")),
        }

    def play_sound(self, action):
        if action in self.sounds.keys():
            pygame.mixer.Sound.play(self.sounds[action])
