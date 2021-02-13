
import numpy as np
import csv


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def save_table(table, key, outname):

    with open(outname, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['name'])
        for val in np.unique(table[key]):
            writer.writerow([val])

def log_edges2centers(edges):
    
    bin_centers = 10**(np.log10(edges[:-1]) + np.diff(np.log10(edges)).mean()/2)

    return bin_centers