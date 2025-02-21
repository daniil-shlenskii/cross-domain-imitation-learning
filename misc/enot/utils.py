import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def mapping_scatter2d(source, target_hat, target):
    fig = plt.figure(figsize=(8, 6))

    plt.scatter(source[:, 0], source[:, 1], color='pink', label='Source')
    plt.scatter(target[:, 0], target[:, 1], color='black', label='Target')
    plt.scatter(target_hat[:, 0], target_hat[:, 1], color='red', label='Target_hat')

    for i in range(len(source)):
        plt.plot([source[i, 0], target_hat[i, 0]], [source[i, 1], target_hat[i, 1]], color='blue')

    plt.legend()
    plt.grid(True)
    plt.close()

    return fig

def mapping_scatter(source, target_hat, target):
    dim = source.shape[-1]

    tsne_embs = np.concatenate([source, target_hat, target])
    if dim > 2:
        tsne_embs = TSNE(n_components=2).fit_transform(tsne_embs)
    source, target_hat, target = (
        tsne_embs[:len(source)],
        tsne_embs[len(source):len(source)+len(target_hat)],
        tsne_embs[len(source)+len(target_hat):]
    )
    return mapping_scatter2d(source, target_hat, target)
