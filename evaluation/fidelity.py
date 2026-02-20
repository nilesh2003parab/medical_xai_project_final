def fidelity_score(original_conf, masked_conf):
    return abs(original_conf - masked_conf)