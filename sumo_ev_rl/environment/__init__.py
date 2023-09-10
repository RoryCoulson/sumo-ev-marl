from gymnasium.envs.registration import register

register(
    id="sumo-ev-rl-v0",
    entry_point="sumo_ev_rl.environment.env:SumoEVEnvironment",
    kwargs={"single_agent": False},
)
