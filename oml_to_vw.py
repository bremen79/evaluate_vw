import argparse
import gzip
import openml
import os
import scipy.sparse as sp
import numpy as np

VW_DS_DIR = './vwdatasets/'

def augment_nom_ft(X, i):
    X_col = X[:,i].astype('str')
    uniq = np.unique(X_col)
    uniq_elts = len(uniq)
    x_len = np.shape(X)[0]
    aug = np.zeros((x_len, uniq_elts))
    for i in range(uniq_elts):
        aug[:,i] = (X_col == uniq[i])
    return aug

def augment_nom_fts(X, categorical_indicator):
    n_features = np.shape(X)[1]
    
    X_aug = []
    for i in range(n_features):
        if categorical_indicator[i]==True:
            X_aug.append(augment_nom_ft(X, i))

    msk_remain = [ True for i in range(np.shape(X)[1]) ]
    for i in range(n_features):
        msk_remain[i] = not categorical_indicator[i]
    X_remain = X[:, msk_remain]
    X_aug.append(X_remain)
    return np.concatenate(X_aug, axis=1)

def save_vw_dataset(X, y, did, ds_dir, n_classes, l2norm):
    if n_classes>2:
        idx_to_labels = range(1,n_classes+1)
    elif n_classes == 2:
        idx_to_labels = [-1, 1]

    fname = 'ds_{}_{}.vw.gz'.format(n_classes, did)
    sparse = sp.isspmatrix_csr(X)

    with gzip.open(os.path.join(ds_dir, fname), 'w') as f:
            for i in range(X.shape[0]):
                if n_classes>=2:
                    f.write('{} '.format(idx_to_labels[y[i]]).encode('utf-8'))
                else:
                    f.write('{} '.format(y[i]).encode('utf-8'))

                if l2norm:
                    norm = np.linalg.norm(X[i])
                else:
                    norm = 1

                if sparse:
                    f.write(' | {}\n'.format(' '.join(
                    '{}:{:.6f}'.format(j, val) for j, val in zip(X[i].indices, X[i].data/norm))).encode('utf-8'))
                else:
                    f.write(' | {}\n'.format(' '.join(
                        '{}:{:.6f}'.format(j, val/norm) for j, val in enumerate(X[i]) if val != 0)).encode('utf-8'))

def shuffle(X, y):
    n = np.shape(X)[0]
    perm = np.random.permutation(n)
    X_shuf = X[perm, :]
    y_shuf = y[perm]
    return X_shuf, y_shuf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openML to vw converter')
    parser.add_argument('--l2norm', action='store_true', help='normalized to L2 norm equal to 1')

    args = parser.parse_args()

    l2norm = args.l2norm

    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory('./omlcache')

    print('loaded openML')

    if not os.path.exists(VW_DS_DIR):
        os.makedirs(VW_DS_DIR)

    openml_list = openml.datasets.list_datasets(output_format='dataframe')
    openml_list = openml_list[openml_list.NumberOfInstances > 20000]
    openml_list = openml_list[openml_list.NumberOfClasses <= 2]
    openml_list = openml_list[openml_list.did > 1112]

    for index, row in openml_list.iterrows():
        did = row['did']
        print('processing did', did)
        try:
            ds = openml.datasets.get_dataset(did)
            X, y, categorical_indicator, attribute_name = ds.get_data(dataset_format='array',target=ds.default_target_attribute)
            X = augment_nom_fts(X, categorical_indicator)
            X, y = shuffle(X, y)
        except Exception as e:
            print(e)
            continue

        save_vw_dataset(X, y, did, VW_DS_DIR, row['NumberOfClasses'], l2norm)
