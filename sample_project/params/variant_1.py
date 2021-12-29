from eparams.eloader import register
from main import Params


@register
def small(params: Params) -> Params:  # no key --> key is function name
    params.training.batch_size = 4
    params.training.num_gpu = 1
    params.training.num_epochs = 4
    return params


@register(key='only cpu')  # key can be any hashable, since it is registered as a dict key
def _(params: Params) -> Params:
    params.training.num_gpu = 0
    params.training.num_cpu = 64
    return params
