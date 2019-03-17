from pysc2.env import sc2_env, run_loop
from pysc2.lib import features

from mini_game_scripted_agents import DefeatZerglingsAndBanelingsRandomAgent

agent = DefeatZerglingsAndBanelingsRandomAgent()

env = sc2_env.SC2Env(
    map_name="FindAndDefeatZerglings",
    players=[sc2_env.Agent(sc2_env.Race.terran)],
    agent_interface_format=features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=84, minimap=64),
        use_feature_units=True),
    step_mul=16,  # the default value is 8, this is also suggested value from the paper
    game_steps_per_episode=0,  # makes the game for as long as necessary
    visualize=False
)

run_loop.run_loop(agent, env)
