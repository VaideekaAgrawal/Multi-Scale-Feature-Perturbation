

#msfp changes here

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
'''
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torchvision.models import ResNet50_Weights

import datasets  # Assuming domainbed.datasets
import hparams_registry
import algorithms
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="MSFP_CCFP")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=1,
        help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="./outputs/train_msfp_2_epochs")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    name = f'out_msfpccfp_{args.trial_seed}_{args.hparams_seed}_{args.dataset}_{args.test_envs}.txt'
    sys.stdout = misc.Tee(os.path.join(args.output_dir, name))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            uda_weights = misc.make_weights_for_balanced_classes(uda) if uda else None
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if uda:
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, env_weights in uda_splits]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = [f'env{i}_in' for i in range(len(in_splits))]
    eval_loader_names += [f'env{i}_out' for i in range(len(out_splits))]
    eval_loader_names += [f'env{i}_uda' for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders) if uda_loaders else None

    train_indices = [i for i in range(len(in_splits)) if i not in args.test_envs]
    steps_per_epoch = sum(len(in_splits[i][0]) for i in train_indices) // hparams['batch_size']
    total_steps = 1 * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    n_steps = args.steps if args.steps is not None else total_steps
    if n_steps > total_steps:
        n_steps = total_steps

    checkpoint_freq = args.checkpoint_freq or 100

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    checkpoint_vals = collections.defaultdict(lambda: [])
    last_results_keys = None

    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x, y in next(train_minibatches_iterator)]
        uda_device = ([x.to(device) for x, _ in next(uda_minibatches_iterator)]
                      if args.task == "domain_adaptation" else None)
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # Fix: Pass name and step to misc.accuracy()
            for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
                acc = misc.accuracy(algorithm, loader, weights, device, name, step)
                results[name + '_acc'] = acc

            results['mem_gb'] = 0.0  # No CUDA

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

'''

'''
import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import gc  # NEW: For memory management
import traceback  # NEW: For detailed error logging

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torchvision.models import ResNet50_Weights

import datasets
import hparams_registry
import algorithms
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="MSFP_CCFP")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=1,
        help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="./outputs/train_msfp_2_epochs")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    name = f'out_msfpccfp_{args.trial_seed}_{args.hparams_seed}_{args.dataset}_{args.test_envs}.txt'
    sys.stdout = misc.Tee(os.path.join(args.output_dir, name))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            uda_weights = misc.make_weights_for_balanced_classes(uda) if uda else None
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if uda:
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, env_weights in uda_splits]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = [f'env{i}_in' for i in range(len(in_splits))]
    eval_loader_names += [f'env{i}_out' for i in range(len(out_splits))]
    eval_loader_names += [f'env{i}_uda' for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders) if uda_loaders else None

    train_indices = [i for i in range(len(in_splits)) if i not in args.test_envs]
    steps_per_epoch = sum(len(in_splits[i][0]) for i in train_indices) // hparams['batch_size']
    total_steps = 1 * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    n_steps = args.steps if args.steps is not None else total_steps
    if n_steps > total_steps:
        n_steps = total_steps

    checkpoint_freq = args.checkpoint_freq or 100

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    checkpoint_vals = collections.defaultdict(lambda: [])
    last_results_keys = None

    # NEW: Wrap training loop in try-except for error handling
    try:
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            
            # NEW: Memory management
            gc.collect()
            torch.cuda.empty_cache()
            
            # NEW: Log GPU memory
            mem_gb = torch.cuda.memory_allocated(device) / 1024**3 if device == "cuda" else 0.0
            print(f"Step {step}, GPU memory: {mem_gb:.2f} GB")
            
            # NEW: Validate minibatches
            try:
                minibatches = next(train_minibatches_iterator)
                for i, (x, y) in enumerate(minibatches):
                    if x.shape[0] != hparams['batch_size'] or x.shape[1:] != dataset.input_shape:
                        raise ValueError(f"Invalid batch {i} shape: {x.shape}, expected [{hparams['batch_size']}, {dataset.input_shape}]")
                    if y.shape[0] != hparams['batch_size'] or not torch.all((y >= 0) & (y < dataset.num_classes)):
                        raise ValueError(f"Invalid label {i} shape: {y.shape} or values: {y}, expected [{hparams['batch_size']}], values in [0, {dataset.num_classes-1}]")
                
                minibatches_device = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                                     for x, y in minibatches]
                
                uda_device = ([x.to(device) for x, _ in next(uda_minibatches_iterator)]
                              if args.task == "domain_adaptation" and uda_minibatches_iterator else None)
                
                # Update model
                step_vals = algorithm.update(minibatches_device, uda_device)
                
                # NEW: Check for NaN/infinite losses
                for key, val in step_vals.items():
                    if torch.isnan(torch.tensor(val)) or torch.isinf(torch.tensor(val)):
                        raise ValueError(f"Invalid loss {key}: {val}")
                
                checkpoint_vals['step_time'].append(time.time() - step_start_time)
                
                for key, val in step_vals.items():
                    checkpoint_vals[key].append(val)
                
                # NEW: Log losses every 10 steps for debugging
                if step % 10 == 0 or step == n_steps - 1:
                    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in step_vals.items()])
                    print(f"Step {step}, Losses: {loss_str}")
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM Error at Step {step}: {str(e)}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
                torch.cuda.empty_cache()
                raise
            except Exception as e:
                print(f"Error at Step {step}: {str(e)}")
                traceback.print_exc()
                raise
            
            if (step % checkpoint_freq == 0) or (step == n_steps - 1):
                results = {
                    'step': step,
                    'epoch': step / steps_per_epoch,
                }

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
                    acc = misc.accuracy(algorithm, loader, weights, device, name, step)
                    results[name + '_acc'] = acc

                results['mem_gb'] = mem_gb  # NEW: Update with actual memory usage

                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys], colwidth=12)

                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })

                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                algorithm_dict = algorithm.state_dict()
                start_step = step + 1
                checkpoint_vals = collections.defaultdict(lambda: [])

                if args.save_model_every_checkpoint:
                    save_checkpoint(f'model_step{step}.pkl')

        save_checkpoint('model.pkl')

        with open(os.path.join(args.output_dir, 'done'), 'w') as f:
            f.write('done')

    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        # NEW: Clean up memory
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Final GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
        '''

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import gc
import traceback
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torchvision.models import ResNet50_Weights
import pandas as pd

import datasets
import hparams_registry
import algorithms
from lib import misc
from lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

def compute_accuracy(logits, labels):
    """Calculate accuracy from logits and labels."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float().sum()
    total = labels.size(0)
    return 100.0 * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="MSFP_CCFP")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=1,
        help='Trial number (used for seeding split_dataset and random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="./outputs/train_msfp_2_epochs")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # Set PyTorch memory allocation config
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    name = f'out_msfpccfp_{args.trial_seed}_{args.hparams_seed}_{args.dataset}_{args.test_envs}.txt'
    sys.stdout = misc.Tee(os.path.join(args.output_dir, name))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Clear GPU memory before loading dataset
    gc.collect()
    torch.cuda.empty_cache()

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []
        out, in_ = misc.split_dataset(env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            uda_weights = misc.make_weights_for_balanced_classes(uda) if uda else None
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if uda:
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, env_weights in uda_splits]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = [f'env{i}_in' for i in range(len(in_splits))]
    eval_loader_names += [f'env{i}_out' for i in range(len(out_splits))]
    eval_loader_names += [f'env{i}_uda' for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders) if uda_loaders else None

    train_indices = [i for i in range(len(in_splits)) if i not in args.test_envs]
    steps_per_epoch = sum(len(in_splits[i][0]) for i in train_indices) // hparams['batch_size']
    total_steps = hparams.get('n_epochs', 2) * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")
    n_steps = args.steps if args.steps is not None else total_steps
    if n_steps > total_steps:
        n_steps = total_steps

    checkpoint_freq = args.checkpoint_freq or 100

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    checkpoint_vals = collections.defaultdict(lambda: [])
    last_results_keys = None
    train_accuracies = []

    # Gradient accumulation setup
    accum_steps = 2
    effective_batch_size = hparams['batch_size'] * accum_steps
    print(f"Using gradient accumulation: batch_size={hparams['batch_size']}, accum_steps={accum_steps}, effective_batch_size={effective_batch_size}")

    try:
        for step in range(start_step, n_steps):
            step_start_time = time.time()
            epoch = step // steps_per_epoch
            
            gc.collect()
            torch.cuda.empty_cache()
            
            mem_gb = torch.cuda.memory_allocated(device) / 1024**3 if device == "cuda" else 0.0
            print(f"Step {step}, GPU memory: {mem_gb:.2f} GB")
            
            try:
                # Gradient accumulation
                total_loss = 0
                step_vals = None
                for acc_step in range(accum_steps):
                    minibatches = next(train_minibatches_iterator)
                    for i, (x, y) in enumerate(minibatches):
                        if x.shape[0] != hparams['batch_size'] or x.shape[1:] != dataset.input_shape:
                            raise ValueError(f"Invalid batch {i} shape: {x.shape} in accum_step {acc_step}")
                        if y.shape[0] != hparams['batch_size'] or not torch.all((y >= 0) & (y < dataset.num_classes)):
                            raise ValueError(f"Invalid label {i} shape: {y.shape} or values: {y} in accum_step {acc_step}")
                    
                    minibatches_device = [(x.to(device, non_blocking=True), y.to(device, non_blocking=True))
                                         for x, y in minibatches]
                    
                    uda_device = ([x.to(device) for x, _ in next(uda_minibatches_iterator)]
                                  if args.task == "domain_adaptation" and uda_minibatches_iterator else None)
                    
                    step_vals = algorithm.update(minibatches_device, uda_device)
                    loss = sum(v for k, v in step_vals.items() if k.startswith('loss_') and not k.startswith('scalar_loss_'))
                    total_loss += loss.item() / accum_steps
                
                # Compute training accuracy
                s_deep_logits = step_vals['s_deep_logits']
                labels = torch.cat([y for _, y in minibatches_device])
                train_acc = compute_accuracy(s_deep_logits, labels)
                checkpoint_vals['train_acc'].append(train_acc.item())
                
                checkpoint_vals['step_time'].append(time.time() - step_start_time)
                
                # Log scalar values only
                for key, val in step_vals.items():
                    if key in ['s_deep_logits', 't_deep_logits']:
                        continue  # Skip non-scalar tensors
                    if isinstance(val, torch.Tensor) and val.numel() == 1:
                        checkpoint_vals[key].append(val.item())
                    elif isinstance(val, (int, float)):
                        checkpoint_vals[key].append(val)
                
                # Print scalar losses and accuracy every 10 steps
                if step % 10 == 0 or step == n_steps - 1:
                    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in step_vals.items() if k.startswith('scalar_')])
                    print(f"Step {step}, Losses: {loss_str}, Train Accuracy: {train_acc:.2f}%")
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM Error at Step {step}: {str(e)}")
                print(f"GPU memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")
                torch.cuda.empty_cache()
                raise
            except Exception as e:
                print(f"Error at Step {step}: {str(e)}")
                traceback.print_exc()
                raise
            
            if (step % checkpoint_freq == 0) or (step == n_steps - 1):
                results = {
                    'step': step,
                    'epoch': step / steps_per_epoch,
                }

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                for name, loader, weights in zip(eval_loader_names, eval_loaders, eval_weights):
                    acc = misc.accuracy(algorithm, loader, weights, device, name, step)
                    results[name + '_acc'] = acc

                results['mem_gb'] = mem_gb

                # Print epoch accuracy
                if (step % steps_per_epoch == 0 and step > 0) or step == n_steps - 1:
                    epoch_acc = np.mean(checkpoint_vals['train_acc'])
                    train_accuracies.append(epoch_acc)
                    print(f"Epoch {int(epoch)}, Train Accuracy: {epoch_acc:.2f}%")
                
                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys], colwidth=12)

                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })

                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")

                # Save summary JSON
                summary = {'train_accuracies': train_accuracies}
                with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
                    json.dump(summary, f, indent=4)
                
                # Display summary table
                if train_accuracies:
                    df = pd.DataFrame({
                        'Epoch': range(1, len(train_accuracies) + 1),
                        'Train Accuracy (%)': [f"{acc:.2f}" for acc in train_accuracies]
                    })
                    print("\nTraining Accuracy Summary:")
                    print(df)

                algorithm_dict = algorithm.state_dict()
                start_step = step + 1
                checkpoint_vals = collections.defaultdict(lambda: [])

                if args.save_model_every_checkpoint:
                    save_checkpoint(f'model_step{step}.pkl')

        save_checkpoint('model.pkl')

        with open(os.path.join(args.output_dir, 'done'), 'w') as f:
            f.write('done')

    except Exception as e:
        print(f"Training failed: {str(e)}")
        traceback.print_exc()
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Final GPU memory: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB")