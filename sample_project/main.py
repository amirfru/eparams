import argparse
from datetime import datetime

import git

import eparams
import eparams.constraints as cc
from eparams import eloader

_repo = git.Repo(__file__, search_parent_directories=True)


@eparams.params(frozen=True)  # frozen means this cannot be changed easily
class Version:
    time = datetime.now()
    git_hash = _repo.head.commit.hexsha
    git_branch = _repo.active_branch.name if not _repo.head.is_detached else 'HEAD detached'
    is_dirty = _repo.is_dirty()
    git_status = _repo.git.status(porcelain=True)


@eparams.params
class TrainingParams:
    experiment_name = 'expr'
    random_seed = 42
    dataset = eparams.Var('/Users/amir', constraints=cc.isdir)
    num_gpu = eparams.Var(8, constraints=cc.in_range(0, 8))
    num_cpu = 36
    batch_size = eparams.Var(64, constraints=lambda n: (n != 0) and (n & (n-1) == 0))  # custom constraint: power of 2
    num_epochs = 30
    learning_rate = [(0, 0.001),
                     (10, 0.0005),
                     (20, 0.00025),
                     (30, 0.000125)]
    weight_decay = 0.001


@eparams.params
class Params:
    output_dir: str  # no default value, must be passed to constructor
    training = TrainingParams()
    version = Version()
    verbose = False

##################################################
#   Example run parameters:
#   python main.py
#   --output_dir=/Users/amir/tmp
#   --delta_configs
#   "only cpu"
#   --params
#   training.experiment_name="new experiment 3"
#   training.batch_size=32
##################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True, help='output dir')
    parser.add_argument('--delta_configs_patterns', default=['params/*.py'], nargs='*',
                        help='glob pattern of delta config files registered via @register')
    parser.add_argument('--delta_configs', default=[], nargs='*', help='list of delta-config keys to apply')
    parser.add_argument('--params', default=[], nargs='*', help='manually set individual params using param.name=val')
    args = parser.parse_args()

    config = Params(output_dir=args.output_dir)  # initialize the config

    # read all .py files in .params/ folder to search for delta configs and save into `eloader.mapping`
    for delta_config_pattern in args.delta_configs_patterns:
        eloader.scan_for_registered('params/*.py')

    # `eloader.mapping is a dictionary of delta-config functions
    for delta_config in args.delta_configs:
        config = eloader.mapping[delta_config](config)

    # eparam's __setitem__ can be used to set a dot-separated param
    for param in args.params:
        param_path, val = param.split('=')
        config[param_path] = val

    print(config)  # print flattened version of the config
    print(config._to_dict())  # config to dict
    print(config._to_yaml())  # config to yaml. config._to_yaml('/path/to/output.yaml') will save to path
