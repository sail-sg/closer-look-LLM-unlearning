import copy
import random

import numpy as np
import torch
import yaml


def get_model_identifiers_from_yaml(model_family):
    # path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]


def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] + value  # or use other logic to merge lists
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    return a_copy


def get_total_len(name, forget_rate):
    if name == 'eval_real_author_wo_options.json':
        return 100
    elif name == 'eval_real_world_wo_options.json':
        return 117
    elif name == 'eval_log.json':
        return 300
    else:
        if forget_rate == 'forget01':
            return 40
        elif forget_rate == 'forget05':
            return 200
        else:
            return 300


def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i + size])
        c.extend(b[i:i + size])
    return c


# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically 
def interleave_eval_result_dict(eval_result_dict, forget_rate, large_bsz, num_processes=2):
    small_bsz = large_bsz // 4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = small_bsz if 'perturb' in metric or 'paraphrase' in metric else large_bsz
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0:len(value) // 2]
            b = value[len(value) // 2:]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
