import argparse
import os
import re
import subprocess
import sys
import time

USE_ADF = True
USE_CS = False

VW = '/home/bremen/Projects/vowpal_wabbit/build/vowpalwabbit/vw'
VW_DS_DIR = './vwdatasets/'
DIR_PATTERN = './res/results_{}/'

rgx = re.compile('^average loss = (.*)$', flags=re.M)

params_dict = {
    'alg': [
        ('sgd'),
        ('coin'),
        ],
    #'loss_function': ['hinge'],
    'loss_function': ['quantile'],
    }
#    'learning_rate': ['0.5'],
#        'ftrl_beta': [1],

extra_flags = None

def param_grid():
    grid = [{}]
    for k in params_dict:
        new_grid = []
        for g in grid:
            for param in params_dict[k]:
                gg = g.copy()
                gg[k] = param
                new_grid.append(gg)
        grid = new_grid

    return grid


def ds_files():
    # only binary datasets
    import glob
    return sorted(glob.glob(os.path.join(VW_DS_DIR, 'ds_0.0_*.vw.gz')))


def get_task_name(ds, params):
    did = int(os.path.basename(ds).split('_')[2].split('.')[0])

    task_name = 'ds:{}'.format(did)
    print(params)
    task_name += '|' + '|'.join('{}:{}'.format(k, v) for k, v in sorted(params.items()) if len(params_dict[k])>1)
    return task_name


def process(ds, params, results_dir):
    print('processing', ds, params)
    did = int(os.path.basename(ds).split('_')[2].split('.')[0])

    #cmd = [VW, ds, '-b', '24', '--ftrl_alpha', '16.0', '--ftrl_beta', '4.0']
    cmd = [VW, ds, '-b', '24']
    for k, v in params.items():
        if k == 'alg':
            if v == 'sgd':
                pass
                #cmd += ['--adaptive']
                #cmd +=['normalized']
            elif v == 'pistol':
                cmd += ['--pistol']
            elif v == 'coin':
                cmd += ['--coin']
        else:
            cmd += ['--{}'.format(k), str(v)]

    print('running', cmd)
    t = time.time()
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8')
    sys.stderr.write('\n\n{}, {}, time: {}, output:\n'.format(ds, params, time.time() - t))
    sys.stderr.write(output)
    pv_loss = float(rgx.findall(output)[0])
    print('elapsed time:', time.time() - t, 'pv loss:', pv_loss)

    return pv_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vw job')
    parser.add_argument('--task_offset', type=int, default=0,
                        help='offset for task_id in output filenames')
    parser.add_argument('--name', default='vw')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--num_ds', default=1000, help='maximum number of datasets to process')
    parser.add_argument('--task_id', default=0, help='first id to process')
    args = parser.parse_args()

    args.results_dir = DIR_PATTERN.format(args.name)

    grid = param_grid()
    dss = ds_files()
    tot_jobs = len(grid) * min(len(dss), int(args.num_ds))

    if args.task_id == 0:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
            import stat
            os.chmod(args.results_dir, os.stat(args.results_dir).st_mode | stat.S_IWOTH)
    else:
        while not os.path.exists(args.results_dir):
            time.sleep(1)
    if not args.test:
        fname = os.path.join(args.results_dir, 'loss{}.txt'.format(args.task_offset + args.task_id))
        done_tasks = set()
        #if os.path.exists(fname):
        #    done_tasks = set([line.split()[0] for line in open(fname).readlines()])
        #loss_file = open(fname, 'a')
        loss_file = open(fname, 'w')
    idx = args.task_id
    while idx < tot_jobs:
        ds = dss[int(idx / len(grid))]
        params = grid[idx % len(grid)]
        if args.test:
            print(ds, params)
        else:
            print(ds, params)
            task_name = get_task_name(ds, params)
            print(task_name)
            if task_name not in done_tasks:
                try:
                    pv_loss = process(ds, params, args.results_dir)
                    loss_file.write('{} {}\n'.format(task_name, pv_loss))
                    loss_file.flush()
                    os.fsync(loss_file.fileno())
                except subprocess.CalledProcessError:
                    sys.stderr.write('\nERROR: TASK FAILED {} {}\n\n'.format(ds, params))
                    print('ERROR: TASK FAILED', ds, params)
        #idx += args.num_tasks
        idx += 1

    if not args.test:
        loss_file.close()
