import sys; args = sys.argv[1:]
#import tensorflow as tf
import numpy as np
from math import sqrt
import math

a = b = 10 ** -3

def inertial_average(tsr, i):
    vec = tsr#tsr.numpy()
    c_len = len(tsr) - tsr.count(0) - 1
    vi = vec[i]
    mu_i = sum(v for idx, v in enumerate(vec) if idx != i)/c_len
    Di = a * (vi - mu_i) + b
    di = Di/(len(vec) - 1)
    vec_new = []
    for idx, v in enumerate(vec):
        if v == 0:
            vec_new.append(0)
        elif idx == i:
            vec_new.append(sqrt(v**2 + Di))
        else:
            vec_new.append(sqrt(v**2 - di))
    tsr_new = vec_new#tf.Tensor(vec_new)
    return tsr_new

def next_vi_ideal(vi, n):
    return sqrt(vi ** 2 + a + (vi - sqrt((1 - vi ** 2) / (n - 1))) * b)

if __name__ == "__main__":
    vec = [0.25, 0.25, 0.25, 0.25]#[0, 0, 0.25, 0, 0.25, 0.25, 0, 0.25, 0, 0, 0]
    tsr = vec#tf.Tensor(vec)
    tsr_new = inertial_average(tsr, 2)
    vec_new = tsr_new#tsr_new.numpy()
    print(vec_new)
