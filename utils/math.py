import numpy as np


def project_a_to_b(a: np.ndarray, b: np.ndarray):
    return scalar_product_fn(a, b) * b / scalar_product_fn(b, b)

def cosine_similarity_fn(a: np.ndarray, b: np.ndarray):
    return scalar_product_fn(a, b) / norm_fn(a) / norm_fn(b) 

def norm_fn(a: np.ndarray):
    return scalar_product_fn(a, a)**0.5

def scalar_product_fn(a: np.ndarray, b: np.ndarray):
    return (a * b).sum(-1)
