import pygame
import enum


class SwitchProController(enum.IntEnum):
    J_A = 0
    J_B = 1
    J_X = 2
    J_Y = 3
    J_MINUS = 4
    J_HOME = 5
    J_PLUS = 6
    J_L3 = 7
    J_R3 = 8
    J_L = 9
    J_R = 10
    J_UP = 11
    J_DOWN = 12
    J_LEFT = 13
    J_RIGHT = 14
    J_SNAP = 15
    J_ZL = enum.auto()
    J_ZR = enum.auto()


def controller_pro_event(event):
    axis_map = {
        4: SwitchProController.J_ZL,
        5: SwitchProController.J_ZR,
    }

    match event.type:
        case pygame.JOYAXISMOTION:
            if event.axis in axis_map.keys() and event.value > 0:
                return axis_map[event.axis]
            return None
        case pygame.JOYBUTTONDOWN:
            return SwitchProController(event.button)

    return None


def main():
    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    pygame.display.set_mode([200, 200])
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            res: SwitchProController = controller_pro_event(event)
            if res is not None:
                print(res.name)

        pygame.display.flip()
        clock.tick(10)


if __name__ == '__main__':
    main()
