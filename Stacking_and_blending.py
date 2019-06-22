import numpy as np
import pandas as pd
import os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sympy 
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso
from sklearn.mixture import GaussianMixture
import scipy
from sklearn.neighbors import KNeighborsClassifier
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.svm import NuSVC
from sklearn import svm, neighbors, linear_model, neural_network
from sklearn.covariance import EmpiricalCovariance
from sklearn.pipeline import Pipeline
import warnings

import multiprocessing
from itertools import combinations
from scipy.optimize import minimize  

import subprocess
import re
import sys
import glob
import ctypes

warnings.filterwarnings('ignore')
#%matplotlib inline

magic = 'wheezy-copper-turtle-magic'


#-----------single threading setting(GMM and QDA optimization)----------------
_MKL_ = 'mkl'
_OPENBLAS_ = 'openblas'


class BLAS:
    def __init__(self, cdll, kind):
        if kind not in (_MKL_, _OPENBLAS_):
            raise ValueError(f'kind must be {MKL} or {OPENBLAS}, got {kind} instead.')
        
        self.kind = kind
        self.cdll = cdll
        
        if kind == _MKL_:
            self.get_n_threads = cdll.MKL_Get_Max_Threads
            self.set_n_threads = cdll.MKL_Set_Num_Threads
        else:
            self.get_n_threads = cdll.openblas_get_num_threads
            self.set_n_threads = cdll.openblas_set_num_threads
            

def get_blas(numpy_module):
    LDD = 'ldd'
    LDD_PATTERN = r'^\t(?P<lib>.*{}.*) => (?P<path>.*) \(0x.*$'

    NUMPY_PATH = os.path.join(numpy_module.__path__[0], 'core')
    MULTIARRAY_PATH = glob.glob(os.path.join(NUMPY_PATH, '_multiarray_umath.*so'))[0]
    ldd_result = subprocess.run(
        args=[LDD, MULTIARRAY_PATH], 
        check=True,
        stdout=subprocess.PIPE, 
        universal_newlines=True
    )

    output = ldd_result.stdout

    if _MKL_ in output:
        kind = _MKL_
    elif _OPENBLAS_ in output:
        kind = _OPENBLAS_
    else:
        return

    pattern = LDD_PATTERN.format(kind)
    match = re.search(pattern, output, flags=re.MULTILINE)

    if match:
        lib = ctypes.CDLL(match.groupdict()['path'])
        return BLAS(lib, kind)
    

class single_threaded:
    def __init__(self, numpy_module=None):
        if numpy_module is not None:
            self.blas = get_blas(numpy_module)
        else:
            import numpy
            self.blas = get_blas(numpy)

    def __enter__(self):
        if self.blas is not None:
            self.old_n_threads = self.blas.get_n_threads()
            self.blas.set_n_threads(1)
        else:
            warnings.warn(
                'No MKL/OpenBLAS found, assuming NumPy is single-threaded.'
            )

    def __exit__(self, *args):
        if self.blas is not None:
            self.blas.set_n_threads(self.old_n_threads)
            if self.blas.get_n_threads() != self.old_n_threads:
                message = (
                    f'Failed to reset {self.blas.kind} '
                    f'to {self.old_n_threads} threads (previous value).'
                )
                raise RuntimeError(message)
    
    def __call__(self, func):
        def _func(*args, **kwargs):
            self.__enter__()
            func_result = func(*args, **kwargs)
            self.__exit__()
            return func_result
        return _func

#------------single threading setting end----------------



def QDA_prediction(train, test, seed=42):
    cols = [c for c in train.columns if c not in ['id', 'target']]
    cols.remove('wheezy-copper-turtle-magic')
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    for i in tqdm_notebook(range(512)):

        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
        pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
        data2 = pipe.fit_transform(data[cols])
        train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

        for r in range(30):
            skf = StratifiedKFold(n_splits=10, random_state=42+r, shuffle=True)
            for train_index, test_index in skf.split(train2, train2['target']):

                clf = QuadraticDiscriminantAnalysis(0.5)
                clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
                oof[idx1[test_index]] += clf.predict_proba(train3[test_index,:])[:,1] / 30.0
                preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits / 30.0

    auc = roc_auc_score(train['target'], oof)
    print(f'AUC: {auc:.5}')
    result_array = []
    for itr in range(4):
        test['target'] = preds
        test.loc[test['target'] > 0.955, 'target'] = 1
        test.loc[test['target'] < 0.045, 'target'] = 0
        usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]
        new_train = pd.concat([train, usefull_test]).reset_index(drop=True)
        print(usefull_test.shape[0], "Test Records added for iteration : ", itr)
        new_train.loc[oof > 0.995, 'target'] = 1
        new_train.loc[oof < 0.005, 'target'] = 0
        oof2 = np.zeros(len(train))
        preds = np.zeros(len(test))
        for i in tqdm_notebook(range(512)):

            train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]
            test2 = test[test['wheezy-copper-turtle-magic']==i]
            idx1 = train[train['wheezy-copper-turtle-magic']==i].index
            idx2 = test2.index
            train2.reset_index(drop=True,inplace=True)

            data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
            pipe = Pipeline([('vt', VarianceThreshold(threshold=2)), ('scaler', StandardScaler())])
            data2 = pipe.fit_transform(data[cols])
            train3 = data2[:train2.shape[0]]
            test3 = data2[train2.shape[0]:]


            random_seed_num = 30
            for r in range(random_seed_num):
                skf = StratifiedKFold(n_splits=10, random_state=seed+r, shuffle=True)
                for train_index, test_index in skf.split(train2, train2['target']):
                    oof_test_index = [t for t in test_index if t < len(idx1)]

                    clf = QuadraticDiscriminantAnalysis(0.5)
                    clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
                    if len(oof_test_index) > 0:
                        oof2[idx1[oof_test_index]] += clf.predict_proba(train3[oof_test_index,:])[:,1] / random_seed_num
                    preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits / random_seed_num

        
        result_array.append([oof2, preds])
        auc = roc_auc_score(train['target'], oof2)
        print(f'AUC: {auc:.5}')
    return result_array



def GMM_prediction(train, test, target_magic=None, seed=42, trained_parameter_file=None):
    if target_magic is not None:
        train = train[train[magic] == target_magic]
        test = test[test[magic] == target_magic]
        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)
    
    if trained_parameter_file is not None:
        trained_parameter = dict(np.load(trained_parameter_file))
        # trained_parameter = np.load(trained_parameter_file)
    else:
        trained_parameter = {}
    
    def get_mean_cov(x,y):
        max_label = y.astype(int).max()
        
        ps = []
        ms = []
        
        for i in range(max_label + 1):
        
            model = GraphicalLasso()
            label_i = (y==i).astype(bool)
            x2 = x[label_i]
            
            model.fit(x2)
            ps.append(model.precision_)
            ms.append(model.location_)

        ms = np.stack(ms)
        ps = np.stack(ps)
        
        return ms,ps
    
    # INITIALIZE VARIABLES
    cols = [c for c in train.columns if c not in ['id', 'target']]
    cols.remove('wheezy-copper-turtle-magic')
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    
    # BUILD 512 SEPARATE MODELS
    random_seed_num = 8
    GMM_array = []
    for r in range(random_seed_num):
        GMM_array.append([np.zeros(len(train)), np.zeros(len(test))])
        
    for i in tqdm_notebook(range(512) if target_magic is None else [target_magic]):
        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        # FEATURE SELECTION 
        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        k = 3 # cluster_per_class
        
        for r in range(random_seed_num):
            # Initialize
#             oof = np.zeros(len(train))
#             preds = np.zeros(len(test))

            # STRATIFIED K-FOLD
            skf = StratifiedKFold(n_splits=11, random_state=seed+r, shuffle=True)
            for j, (train_index, test_index) in enumerate(skf.split(train3, train2['target'])):

                ms_key = "ms_{}_{}_{}".format(i, r, j)
                ps_key = "ps_{}_{}_{}".format(i, r, j)
                
                if ms_key in trained_parameter and ps_key in trained_parameter:
                    ms = trained_parameter[ms_key]
                    ps = trained_parameter[ps_key]
                else:
                    # MODEL AND PREDICT WITH GMM
                    new_label = np.zeros(len(train_index))
                    try_cnt = 0
                    while True:            
                        gm = GaussianMixture(random_state=seed+try_cnt+r, n_components=k).fit(train3[train_index,:][train2.loc[train_index]['target'] == 0])
                        new_label[train2.loc[train_index]['target'] == 0] = gm.predict(train3[train_index,:][train2.loc[train_index]['target'] == 0, :])
                        gm = GaussianMixture(random_state=seed+try_cnt+r, n_components=k).fit(train3[train_index,:][train2.loc[train_index]['target'] == 1])
                        new_label[train2.loc[train_index]['target'] == 1] = k + gm.predict(train3[train_index,:][train2.loc[train_index]['target'] == 1, :])

                        try:
                            ms, ps = get_mean_cov(train3[train_index,:], new_label)
                        except (FloatingPointError,ValueError) as e:
                            try_cnt += 1
                            continue
                        else:
                            break

                gm = GaussianMixture(random_state=seed, n_components=2*k, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)
                gm.fit(np.concatenate([train3[train_index,:], test3, train3[test_index, :]],axis = 0))
                
                GMM_array[r][0][idx1[test_index]] += np.sum(gm.predict_proba(train3[test_index,:])[:,k:], axis=1) 
                GMM_array[r][1][idx2] += np.sum(gm.predict_proba(test3)[:,k:], axis=1) / skf.n_splits
#                 oof[idx1[test_index]] += np.sum(gm.predict_proba(train3[test_index,:])[:,k:], axis=1) #/ random_seed_num
#                 preds[idx2] += np.sum(gm.predict_proba(test3)[:,k:], axis=1) / skf.n_splits #/ random_seed_num
#             GMM_array.append([oof, preds])

    # Print cv GMM
    averaging_oof = np.zeros(len(train))
    for array in GMM_array:
        averaging_oof += (array[0] / random_seed_num)
    auc = roc_auc_score(train['target'],averaging_oof)
    print('GMM_random_seed_averaging CV =',round(auc,5))
    
    return GMM_array




if __name__=='__main__':
    train = pd.read_csv('../input/instant-gratification/train.csv')
    test = pd.read_csv('../input/instant-gratification/test.csv')
    with single_threaded(np):
        QDA_array = QDA_prediction(train, test)
        GMM_array = solution(train.copy(), test.copy(), trained_parameter_file='../input/instanttrainedparameter/trained_parameter_k_2_r_10_j_11.npz')

    # Stacking
    for i, array in enumerate(GMM_array):
        # Initialization
        if i==0:
            tr = array[0].reshape(-1, 1)
            te = array[1].reshape(-1, 1)
            continue
        tr = np.concatenate((tr, array[0].reshape(-1, 1)), axis=1)
        te = np.concatenate((te, array[1].reshape(-1, 1)), axis=1)
        
    for i, array in enumerate(QDA_array):
        tr = np.concatenate((tr, array[0].reshape(-1, 1)), axis=1)
        te = np.concatenate((te, array[1].reshape(-1, 1)), axis=1)
        
    oof_lrr = np.zeros(len(train))
    pred_lrr = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(tr, train['target']):
        lrr = linear_model.LogisticRegression() 
        lrr.fit(tr[train_index], train['target'][train_index])
        oof_lrr[test_index] = lrr.predict_proba(tr[test_index,:])[:,1]
        pred_lrr += lrr.predict_proba(te)[:,1] / skf.n_splits
        
    auc_lrr = round(roc_auc_score(train['target'],oof_lrr),6)
    print('stack CV score =', auc_lrr)



    # Random seed averaging
    GMM_averaging_oof = np.zeros(len(train))
    GMM_averaging_pred = np.zeros(len(test))
    for array in GMM_array:
        GMM_averaging_oof += (array[0] / len(GMM_array)) 
        GMM_averaging_pred += (array[1] / len(GMM_array))


    # search best blending weight
    best_i = 0
    best_blend_score = 0
    for i in range(101):
        print((1*i), (1*(100-i)), end="")
        auc_blend = roc_auc_score(train['target'],(0.01*i)*GMM_averaging_oof+(0.01*(100-i))*oof_lrr)
        print(': stack + blend CV score =', auc_blend)
        if best_blend_score<auc_blend:
            best_blend_score = auc_blend
            best_i = i


    if auc_lrr<best_blend_score:
        print("stacking+blending adopted: {}".format(best_i/100))
        pred_adopt = (0.01*best_i)*GMM_averaging_pred+(0.01*(100-best_i))*pred_lrr
    else:
        print("only stacking adopted")
        pred_adopt = pred_lrr

    sub = pd.read_csv('../input/instant-gratification/sample_submission.csv')
    sub['target'] = pred_adopt
    sub.to_csv('submission.csv', index=False)

    # prediction histogram
    plt.hist(pred_adopt, bins=100)
    plt.show()
