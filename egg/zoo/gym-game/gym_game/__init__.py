
from gym.envs.registration import register

register(
        id='comm-game-v0',
        entry_point='gym_game.envs:GameEnv',
)
