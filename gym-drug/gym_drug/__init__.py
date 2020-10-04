from gym.envs.registration import register

register(
    id='dd-v0',
    entry_point='gym_drug.envs:StatesEnv',
)
