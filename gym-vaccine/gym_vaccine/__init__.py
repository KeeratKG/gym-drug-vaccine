from gym.envs.registration import register

register(
    id='vd-v0',
    entry_point='gym_vaccine.envs:StatesEnv',
)
