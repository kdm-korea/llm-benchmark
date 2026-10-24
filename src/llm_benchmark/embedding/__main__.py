"""임베딩 실험의 CLI 인자를 읽고 선택한 처리 단계를 실행한다."""

import argparse
import json
import sys

from .config import load_config
from .dataset import download_dataset, prepare_dataset
from .run import probe_models, run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description='Legal embedding exact-retrieval benchmark')
    parser.add_argument('command', choices=('download', 'prepare', 'probe', 'run'))
    parser.add_argument('--config', default='configs/legal_recall.yaml')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--query-limit', type=int)
    parser.add_argument('--device', choices=('auto', 'cpu', 'mps', 'cuda'))
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.device:
            config['runtime']['device'] = args.device
        if args.batch_size is not None:
            if args.batch_size < 1:
                raise ValueError('batch-size must be positive')
            config['runtime']['batch_size'] = args.batch_size

        if args.command == 'download':
            print(json.dumps(download_dataset(config['dataset']), indent=2))
        elif args.command == 'prepare':
            _, report = prepare_dataset(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == 'probe':
            print(probe_models(config, args.models))
        else:
            print(run_benchmark(config, args.models, args.query_limit))
    except (ValueError, OSError, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
