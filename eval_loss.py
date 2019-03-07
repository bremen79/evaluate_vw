import argparse
import os
import re
import sys
import math
import scipy.stats
import gzip

import numpy as np
import pandas as pd

USE_ADF = True
USE_CS = False

DIR_PATTERN = './cb_eval/res/cbresults_{}/'

def load_raw(loss_file):
    rgx = re.compile(r'^ds:(\d+)\|alg:(.*) (.*)$', flags=re.M)
    if loss_file.endswith('.gz'):
        lines = rgx.findall(gzip.open(loss_file).read())
    else:
        lines = rgx.findall(open(loss_file).read())
    df_raw = pd.DataFrame(lines, columns=['ds', 'alg', 'loss'])

    df_raw.ds = df_raw.ds.astype(int)
    df_raw.loss = df_raw.loss.astype(float)

    return df_raw



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval losses')
    parser.add_argument('--results_dir', default='./res/results_vw/')
    args = parser.parse_args()

    loss_file = os.path.join(args.results_dir, 'all_losses.txt')
    if os.path.exists(loss_file + '.gz'): # prefer gzipped file if it's there
        loss_file += '.gz'
    if not os.path.exists(loss_file):
        sys.stderr.write('concatenating loss files...')
        import subprocess
        subprocess.check_call(['cat {} | sort > {}'.format(os.path.join(args.results_dir, 'loss*.txt'),
                               loss_file)], shell=True)

    df = load_raw(loss_file)
    
    kt_res=df.loc[df['alg'] == 'kt']
    kt_res.set_index('ds', inplace=True)
    
    sgd_res=df.loc[df['alg'] == 'sgd']
    sgd_res.set_index('ds', inplace=True)
    
    diff=sgd_res.loss.sub(kt_res.loss)
    
    N=len(diff.index)
    
    #print diff
    print 'KT wins ' + repr(len(diff.loc[diff>0].index)) + ' times on ' + repr(N) + ' datasets'
    #rank=diff.abs().rank()
    ##print rank
    #T=min(rank.loc[diff>0].sum(),rank.loc[diff<0].sum())
    #print T
    #print (T-0.25*N*(N+1))/math.sqrt(1.0/24.0*N*(N+1)*(2*N+1))
    print scipy.stats.wilcoxon(diff)
