import enum


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


roll_actions = [
    Action.RIGHT,
    Action.LEFT,
    Action.UP,
    Action.DOWN,
]
rot_actions = [
    Action.ROT_RIGHT,
    Action.ROT_LEFT,
]
save_actions = [
    Action.SAVE1,
    Action.SAVE2,
    Action.SAVE3,
]
load_actions = [
    Action.LOAD1,
    Action.LOAD2,
    Action.LOAD3,
]
memory_actions = save_actions + load_actions
