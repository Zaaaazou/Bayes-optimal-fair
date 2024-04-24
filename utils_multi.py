import numpy as np
import traceback

def cal_acc_multiclass(eta, Y, Z, t, Z_number):
    Yhat = np.zeros_like(eta)
    for z in range(Z_number):
        Yhat[Z == z] = (eta[Z == z]> t[z])
        acc = (Yhat == Y).mean()
        return acc
    

def cal_disparity_multiclass(eta, Z, t, unique_Z):
    disparities = {}
    for z in unique_Z:
        for w in unique_Z:
            if w != z:
                disparities[f"{int(z)}{int(w)}"] = (eta[Z == z]>t[z]).mean() - (eta[Z == w] >t[w]).mean()
    disparity = 1/len(unique_Z) * sum([np.abs(disparities[f"{int(z)}{int(w)}"]) for z in unique_Z for w in unique_Z if w != z])

def cal_t_bound_multiclass(probabilities, Z_values):
    marginal_y = []
    for z in range(len(Z_values)):
        key_0 = f"{int(z)}{int(0)}"
        key_1 = f"{int(z)}{int(1)}"
        marginal_y.append((probabilities[key_0]+probabilities[key_1]))
    upperbound = min(marginal_y)
    lowerbound = -1 * upperbound
    return lowerbound, upperbound




def number_of_sample_multiclass(subset_sizes, Z, t, n_sample):
    probabilities = {}
    samples = {}
    marginal_y = {}
    subset_sizes_new = {}
    Z_unique = Z.unique()
    n = sum(subset_sizes[key] for key in subset_sizes.keys())
    probabilities_new = {}
    for z in len(Z_unique):
        key_0 = f"{int(z)}{int(0)}"
        key_1 = f"{int(z)}{int(1)}"
        probabilities[key_0], probabilities[key_1] = subset_sizes[key_0]/n, subset_sizes[key_1]/n
        marginal_y[z] = probabilities[key_0]+probabilities[key_1]
        samples[key_0], samples[key_1] = probabilities[key_0] * (1 / 2 - t / 2 / marginal_y[z]), probabilities[key_1] * (1 / 2 + t / 2 / marginal_y[z])

    for z in range(len(Z_unique)):
        key_0 = f"{int(z)}{int(0)}"
        key_1 = f"{int(z)}{int(1)}"
        if samples[key_0] < 0:
            samples[key_0] = 0
        if samples[key_1] < 0:
            samples[key_1] = 0
        probabilities_new[key_0] = 0.5 * samples[key_0] *(samples[key_0]+samples[key_1])
        probabilities_new[key_1] = 0.5 * samples[key_1] *(samples[key_0]+samples[key_1])

        subset_sizes_new[key_0] = round(n_sample * probabilities_new[key_0])
        subset_sizes_new[key_1] = round(n_sample * probabilities_new[key_1])
    return subset_sizes_new