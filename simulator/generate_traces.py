import argparse
import json
import numpy as np
import os
args = None

def _parse_args():
    parser = argparse.ArgumentParser(description='Trace generator')

    parser.add_argument('--num-models', type=int, default=6,
                        help='number of different models')
    parser.add_argument('--num-demands', type=int, default=3,
                        help='number of different demands')
    parser.add_argument('--num-jobs', type=int, default=5,
                        help='number of jobs')
    parser.add_argument('--job-trace', default=f'{os.environ["DLCM_PATH"]}/simulator/traces/generated_trace', type=str, metavar='PATH',
                        help='path to the generated job trace file')

    return parser.parse_args()

def main(args):
    print(f'args.num_models {args.num_models} args.num_demands {args.num_demands}')

    models = ['MobileNetV3', 'GraphSage', 'FSDP', 'Transformer-XL', 'DLRM', 'MoE']
    demands = [2, 4, 8]
    total_trained_samples = {
        'MobileNetV3': 1230076.332,
        'GraphSage': 145710106.7,
        'FSDP': 7795.404,
        'Transformer-XL': 48751.95466,
        'DLRM': 17005730.5,
        'MoE': 9945.316645
    }

    per_model_ratio = np.squeeze(0.01*(np.random.multinomial(100, [1/float(args.num_models)]*args.num_models, size=1)))
    per_demands_ratio = np.squeeze(0.01*(np.random.multinomial(100, [1/float(args.num_demands)]*args.num_demands, size=1)))

    print(f'per-model ratio of {models}: {per_model_ratio}')
    print(f'per-demand ratio of {demands}: {per_demands_ratio}')

    # [{"id": "1", "model": "MobileNetV3", "gpu_demand": 2, "total_samples_to_train": 1230076.332, "trained_samples": 0},

    with open(args.job_trace, mode='w') as f:
        print(f'{args.job_trace}')
        trace_dict = {}
        for job_id in range(0, args.num_jobs):
            model = np.random.choice(models, 1, p=per_model_ratio)[0]
            demand = int(np.random.choice(demands, 1, p=per_demands_ratio)[0])
            print(f'{model}-{demand} selected')
            trace_dict[job_id] = {'model': model, 'gpu_demand': demand, 'total_samples_to_train': total_trained_samples[model]}
        print(f'trace_dict {trace_dict}')
        json.dump(trace_dict, f)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
