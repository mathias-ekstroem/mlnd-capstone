def unit_type_is_selected(self, obs, unit_type):
    if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit_type:
        return True

    if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit_type:
        return True

    return False

def get_units_by_type(self, obs, unit_type):
    return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

def can_do(self, obs, action):
    return action in obs.observation.available_actions