import json
import os
import numpy as np
import matplotlib.pyplot as plt

import argparse

# read results dir from command line
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', '-r', type=str, 
                    default='work_dirs',
                    help='path to results directory')
parser.add_argument('--top_only', '-t', action='store_true',
                    help='only find experiments in top-level directories')
args = parser.parse_args()

results_dir = args.results_dir
keys = [
    "ft-",   # Fine-tuning
    "st",    # Separate training
    "jt-",   # Joint training
    "ewc-",  # EWC (Online and Separate)
    "lfl-",  # LFL
    "lwf-",  # LwF
    "iwd-",  # IWD
    "lgf-",  # Ablations of IWD
]

def find_results(results_dir, keys):
    scalars_files = []
    exp_scalars = {}
    experiments = os.listdir(results_dir)
    for exp in experiments:
        for root, _, files in os.walk(os.path.join(results_dir, exp)):
            if all(key not in root for key in keys):
                continue

            if 'scalars.json' in files:
                # Get experiment directory (two levels above root)
                exp_dir = os.path.dirname(os.path.dirname(root.rstrip('/')))

                # If top_only is set, only consider top-level directories
                if args.top_only and os.path.dirname(exp_dir) != results_dir.rstrip('/'):
                    continue

                if exp_dir not in exp_scalars:
                    exp_scalars[exp_dir] = []

                exp_scalars[exp_dir].append(os.path.join(root, 'scalars.json'))

    # Sort all experiment scalars by name and get the last one
    for exp, scalars in exp_scalars.items():
        scalars = sorted(scalars)
        if 'jt-' in exp:
            print(exp)
        scalars_files.append(scalars[-1])

    return scalars_files

def read_results(file):
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
            data = None
            for line in reversed(lines):
                if "coco/AP" in line or "mpii/PCK" in line or "crowdpose/AP" in line:
                    data = json.loads(line)
                    break
            if data is None:
                return None

            name = file.split('/')[-4].split('0e_')[-1]

            exp_num = 0
            if "coco" in name:
                exp_num += 1
            if "mpii" in name:
                exp_num += 1
            if "crowdpose" in name:
                exp_num += 1

            name = '-'.join(name.split('-')[:-exp_num])

            # Parse trial number
            parts = name.split('-')
            trial = 1
            if parts[-1].startswith('trial'):
                name = '-'.join(parts[:-1])
                trial = int(parts[-1].split('trial')[-1])

            data = {
                'experiment': name,
                'trial': trial,
                'task_1': data.get("coco/AP", -0.01) * 100,
                'task_2': data.get("mpii/PCK", -1),
                'task_3': data.get("crowdpose/AP", -0.01) * 100,
                'epoch': data['step'],
            }
            mean = (data['task_1'] + data['task_2'] + data['task_3']) / 3
            data['mean'] = mean

            status = 'OK'
            if data['task_1'] < 0 or data['task_2'] < 0 or data['task_3'] < 0:
                status = 'FAIL'
            data['status'] = status

            return data
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file}")
    except FileNotFoundError:
        print(f"File not found: {file}")
    return None

def summarize_results(scalars_files):
    results = []
    for file in scalars_files:
        r = read_results(file)
        if r is not None:
            results.append(r)

    grouped_results = {}
    for res in results:
        exp = res['experiment']
        if exp not in grouped_results:
            grouped_results[exp] = []
        grouped_results[exp].append(res)

    MAX_TRIALS = 5

    summary = []
    for exp, trials in grouped_results.items():
        # Choose best MAX_TRIALS trials
        trials = sorted(trials, key=lambda x: x['mean'], reverse=True)[:MAX_TRIALS]

        task_1_scores = [trial['task_1'] for trial in trials]
        task_2_scores = [trial['task_2'] for trial in trials]
        task_3_scores = [trial['task_3'] for trial in trials]
        means = [trial['mean'] for trial in trials]

        task_1_mean = np.mean(task_1_scores)
        task_1_std = np.std(task_1_scores)
        task_2_mean = np.mean(task_2_scores)
        task_2_std = np.std(task_2_scores)
        task_3_mean = np.mean(task_3_scores)
        task_3_std = np.std(task_3_scores)
        overall_mean = np.mean(means)
        overall_std = np.std(means)

        # Get mean epoch
        epochs = [trial['epoch'] for trial in trials]
        mean_epochs = int(np.sum(epochs) / MAX_TRIALS)

        status = f'{mean_epochs * 2}% done ({len(trials)}/5)'

        summary.append({
            'experiment': exp,
            'task_1_mean': task_1_mean,
            'task_1_std': task_1_std,
            'task_2_mean': task_2_mean,
            'task_2_std': task_2_std,
            'task_3_mean': task_3_mean,
            'task_3_std': task_3_std,
            'overall_mean': overall_mean,
            'overall_std': overall_std,
            'epoch': mean_epochs,
            'status': status
        })
    return summary

def print_summary(summary):
    # sort by name
    summary = sorted(summary, key=lambda x: x['experiment'], reverse=False)

    print('Experiment           Task 1            Task 2            Task 3            Overall')
    print('----------------------------------------------------------------------------------')
    for res in summary:
        print(f'{res["experiment"]:<20}{res["task_1_mean"]:>6.2f}±{res["task_1_std"]:>3.2f}       {res["task_2_mean"]:>6.2f}±{res["task_2_std"]:>3.2f}       {res["task_3_mean"]:>6.2f}±{res["task_3_std"]:>3.2f}       {res["overall_mean"]:>6.2f}±{res["overall_std"]:>3.2f}')

def main():
    scalars_files = find_results(results_dir, keys)
    summary = summarize_results(scalars_files)
    print_summary(summary)

if __name__ == '__main__':
    main()
