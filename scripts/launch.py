import argparse
import logging
import pathlib
import queue
import subprocess
import sys
import threading

import torch
import yaml


logger = logging.getLogger(__name__)


def load_job_queue(fname):
    q = queue.Queue()
    with open(fname, 'r') as f:
        jobs = yaml.load_all(f)
        for job in jobs:
            logger.info(job)
            q.put(job)
    return q


def main(args):
    logger.info('Loading jobs from: %s', args.input)
    q = load_job_queue(args.input)

    if not args.logdir.exists():
        logger.info('Creating directory: %s', args.logdir)
        args.logdir.mkdir(parents=True)

    log_lock = threading.Lock()

    def worker(rank):
        while True:
            if q.empty():
                return
            # Get job from queue
            job = q.get()

            # Format command, add rank argument
            cmd = [sys.executable, '-u']
            cmd.append(job['script'])
            cmd.append(f'--local_rank={rank}')
            cmd.extend(job['args'])

            stdout = open(args.logdir / f'{job["out"]}.stdout', 'w')
            stderr = open(args.logdir / f'{job["out"]}.stderr', 'w')


            process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
            process.wait()

    # Start all of the jobs
    threads = []
    logger.info('Starting threads')
    for rank in range(torch.cuda.device_count()):
        t = threading.Thread(target=worker, args=(rank,))
        t.start()
        threads.append(t)

    # Wait for them to finish
    for t in threads:
        t.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path, help='JSONL file containing jobs.')
    parser.add_argument('--logdir', type=pathlib.Path, default='results/')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

