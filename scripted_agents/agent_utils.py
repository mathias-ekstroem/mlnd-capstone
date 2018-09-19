import random


def unit_type_is_selected(obs, unit_type):
    if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type:
        return True

    if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type:
        return True

    return False


def get_units_by_type(obs, unit_type):
    return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]


def get_random_unit_by_type(obs, unit_type):
    units = [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    unit = random.choice(units)
    return unit


def can_do(obs, action):
    return action in obs.observation.available_actions
