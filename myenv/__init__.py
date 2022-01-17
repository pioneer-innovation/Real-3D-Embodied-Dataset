from gym.envs.registration import register
register(
    id='R3AD-v0',
    entry_point='myenv.R3AD:R3AD',
)
register(
    id='R3AD-v1',
    entry_point='myenv.R3AD_V1:R3AD_V1',
)
register(
    id='R3AD-v2',
    entry_point='myenv.R3AD_V2:R3AD_V2',
)
register(
    id='R3AD-v3',
    entry_point='myenv.R3AD_V3:R3AD_V3',
)