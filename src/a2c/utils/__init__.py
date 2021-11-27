from utils.additional_torch import to_device, get_flat_grad_from, set_flat_params_to, get_flat_params_from, compute_flat_grad
from utils.math import normal_entropy, normal_log_density, set_init
from utils.replay_memory import Memory
from utils.tools import assets_dir
from zfilter import RunningStat