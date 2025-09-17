"""Preprocessing functions for optical data. Example:
from neurozip.types.nzloader import NzLoad
import aspis.common as A

def dff(a, b, c, data):
    return data * a + b

dff_curried = A.curry(dff)

preprocessor_list  = [
    dff_curried(a, b, c)
]

data = data
for preprocessor in preprocessor_list:
    data = preprocessor(data)

"""
