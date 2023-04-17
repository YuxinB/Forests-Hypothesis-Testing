import numpy as np
from scipy.stats import entropy
from joblib import Parallel, delayed

from hyppo.ksample._utils import k_sample_transform
from hyppo.tools import rot_ksamp
from hyppo.tools import SIMULATIONS
from honest_forests import HonestForestClassifier
from math import ceil

def mutual_info(clf, X, y):
    clf = clf.fit(X, y)
    ### change x to X
    # if clf.honest_prior == 'ignore':
    #       prob = clf.predict_proba(X)[~np.isnan(clf.predict_proba(X)).any(axis=1)]
    #       H_YX = np.mean(entropy(prob, base=np.exp(1), axis=1))
    # else:
    H_YX = np.mean(entropy(clf.predict_proba(X), base=np.exp(1), axis=1))
    
    _, counts = np.unique(y, return_counts=True)
    H_Y = entropy(counts, base=np.exp(1))
    return max(H_Y - H_YX, 0)



def _perm_stat(sim,sample_size = 100,angle = 90,dim = 2): 
    
    # if hasattr(calc_stat, "fit_"):
    #     calc_stat.fit_ = False
    
    x, y= rot_ksamp(sim, n=sample_size,k=2, p=dim,degree = angle, noise=False)
    #print(np.var(y))
    if np.var(y) == 0:
        observe_stat = 0
        null_dist = 0
    else:
        X, y = k_sample_transform([x, y],test_type = 'rf')
        #print(X.shape)
        #print(y.shape)
        #y = np.ravel(y.astype(int))

        observe_stat = mutual_info(clf,X, y)
        # Refit Forests at each permutation
        permy = np.random.permutation(y)
        
        # if hasattr(calc_stat, "fit_"):
        #     calc_stat.fit_ = False
        
        null_dist = mutual_info(clf,X, permy)
    #print(observe_stat,null_dist)
    
    return observe_stat,null_dist


def estimate_power(sim,sample_size_power = 100,angle_power = 90 ,dim_power = 2,reps = 1000): 
    alt_dist, null_dist = map(np.float64,
                              zip(*[_perm_stat(sim,sample_size_power,angle_power,dim_power) for _ in range(reps)]),)
    
    # Obtain power by cutoff
    cutoff = np.sort(null_dist)[ceil(reps * (1 - 0.05))]
    print("cutoff " + str(cutoff))
    empirical_power = (1 + (alt_dist >= cutoff).sum()) / (1 + reps)
    print('power ' + str(empirical_power))
    return empirical_power

#### DIMENSION
def est_power_dim (sim):
        print(sim)
        POWER = []
        for dim_i in dim:
                print(sim,dim_i)
                # Calculate the mean power over 5 times
                power= np.mean([estimate_power(sim,100,90,dim_power = int(dim_i),reps = 1000) for _ in range(power_reps)])
                #print(power)
                POWER.append(power)
        np.save("/home/azureuser/FOREST_HYPOTHESIS_38/OUTPUT/Dim/POWER_DIM_empirical_2sample_0.6_{}.npy".format(sim),
        POWER)
        print(POWER)
        return POWER

#### SampleSize
def est_power_sample(sim):
        POWER = []
        print(sim)
        for size_i in sample_size:
                print(sim,size_i)
                # Calculate the mean power over 5 times
                power= np.mean([estimate_power(sim,int(size_i),90,1,reps = 1000) for _ in range(power_reps)])
                #print(power)
                POWER.append(power)
        np.save("/home/azureuser/FOREST_HYPOTHESIS_38/OUTPUT/SampleSize/POWER_SampleSize_empirical_2sample_0.6_{}.npy".format(sim),
        POWER)
        print(POWER)
        return POWER


### ANGLE
def est_power_angle(sim):
        POWER = []
        print(sim)
        for angle_i in angle:
                print(sim,angle_i)
                # Calculate the mean power over 5 times
                power= np.mean([estimate_power(sim,100,angle_i,1,reps = 1000) for _ in range(power_reps)])
                #print(power)
                POWER.append(power)
        np.save("/home/azureuser/FOREST_HYPOTHESIS_38/OUTPUT/Angle/POWER_Angle_empirical_2sample_0.6_{}.npy".format(sim),
        POWER)
        print(POWER)
        return POWER

clf = HonestForestClassifier(
    n_estimators=100,      # number of trees
    honest_fraction=0.6,   # fraction of samples used to construct the tree
    honest_prior="empirical", # ignore finite sample correction
    #max_features=1,        # consider 1 feature at each split
#     n_jobs=-1,             # run parallel on all threads
)


power_reps =5
#sample_size = 100
dim = np.arange(1,11,1)
sample_size = range(5,105,5)
angle = range(0,95,5)
print(SIMULATIONS.keys())
# outputs_sample = Parallel(n_jobs= -1,verbose = 100)([delayed(est_power_sample)(sim) for sim in list(SIMULATIONS.keys())])
# outputs_angle = Parallel(n_jobs= -1,verbose = 100)([delayed(est_power_angle)(sim) for sim in list(SIMULATIONS.keys())])
# outputs_dim = Parallel(n_jobs= -1,verbose = 100)([delayed(est_power_dim)(sim) for sim in list(SIMULATIONS.keys())])

# SIM = [
#       #'linear', 
#       'square'
# # ,'circle','multimodal_independence'
# ]
#print(SIM)
print('SampleSize')
outputs_sample = Parallel(n_jobs= -1,verbose = 100)([delayed(est_power_sample)(sim) for sim in SIMULATIONS.keys()])
print('Angle')
outputs_angle = Parallel(n_jobs= -1,verbose = 100)([delayed(est_power_angle)(sim) for sim in SIMULATIONS.keys()])
print('Dimension')
outputs_dim = Parallel(n_jobs= -1,verbose = 100)([delayed(est_power_dim)(sim) for sim in SIMULATIONS.keys()])
