import numpy as np
import matplotlib.pyplot as plt
import pylab
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cosine

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SEED = 42


def plot_pca(spectra_vecs, required_keys, class_decoder, name_decoder, colormapper, dim_red='PCA',
             min_bin=0, max_bin=850, figsize=(12, 10), IsNameDecoder=False, path=None, show=True):
    """
    Plots PCA or t-SNE graph.

    Parameters
    ----------
    spectra_vecs : dict
        Dictionary, where keys are names of the spectra (typically mzXML file names) and values are numpy arrays
        with vector components.
    required_keys : list
        List of keys of spectra that should be plotted.
    class_decoder : list
        List of spectra decoders by class.
    name_decoder : list
        List of spectra decoders by name. Use strings.
    colormapper : dict
        Dictionary, which gives each class a unique color.
    dim_red : str
        Dimensionality reduction technique. Can be 'PCA' or 'TSNE'. (default is 'PCA').
    min_bin : int
        Number of the first component of spectra that go through PCA. (default is 0).
    max_bin : int
        Number of the last component of spectra that go through PCA. (default is 850)
    figsize : tuple
        Plot size. (default is (12, 10)).
    IsNameDecoder : bool
        If True name decoders are shown in the plot.
    path : str
        Path to the file, if you want to save graph.

    Raises
    -------
    ValueError
    Incorrect decoder list sizes.
    """

    if len(class_decoder) != len(name_decoder):
        raise ValueError("Decoder lists don't have the same length.")

    X = np.zeros((len(required_keys), max_bin - min_bin))
    i = 0
    for key in required_keys:
        X[i] = spectra_vecs[key][min_bin:max_bin]
        i += 1
    if dim_red == 'PCA':
        pca = PCA(n_components=2, random_state=SEED).fit(X)
        X_reduc = pca.transform(X)
        PC1_coords = X_reduc[:, 0]
        PC2_coords = X_reduc[:, 1]
    elif dim_red == 'TSNE':
        X_reduc = TSNE(n_components=2).fit_transform(X)
        PC1_coords = X_reduc[:, 0]
        PC2_coords = X_reduc[:, 1]
    else:
        raise ValueError('Dimensionality reduction technique is PCA or t-SNE only')

    plt.figure(figsize=figsize)
    for color_name, color in colormapper.items():
        plt.scatter(PC1_coords[class_decoder == color_name], PC2_coords[class_decoder == color_name], c=color,
                    label=color_name)
    if IsNameDecoder:
        for i in range(len(X_reduc)):
            plt.annotate(name_decoder[i], (X_reduc[i][0], X_reduc[i][1]))

    plt.legend()
    plt.xticks([], [])
    plt.yticks([], [])

    plt.xlabel('PC-1')
    plt.ylabel('PC-2')

    if path:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def agglosamples(spectra_vecs, filenames, decoder, show=False, path=None):
    """
    Plots dendrogram based on samples. Used to analyze similarities in spectra.

    Parameters
    ----------
    spectra_vecs : dict
        Dictionary, where keys are names of the spectra (typically mzXML file names) and values are numpy arrays
        with vector components.

    filenames : list
        List with the filenames of required spectra.

    decoder : list
        List with decoders, which use to characterize spectra (for example, "Green", "Black" tea spectra)

    path : str
        Path to the file, if you want to save dendrogram.

    Raises
    -------
    ValueError
    Incorrect decoder list sizes.
    """

    if len(filenames) != len(decoder):
        raise ValueError('Filenames and decoder should have equal length')

    x = np.zeros((len(filenames), len(spectra_vecs[filenames[0]])))
    for i in range(len(filenames)):
        x[i] = spectra_vecs[filenames[i]]

    D = np.zeros([len(filenames), len(filenames)])
    for i in range(len(filenames)):
        for j in range(len(filenames)):
            D[i, j] = cosine(x[i], x[j])

    fig = pylab.figure(figsize=(8, 5))
    axdendro = fig.add_axes([0.09, 0.1, 0.2, 0.8])
    link = linkage(scipy.spatial.distance.squareform(D), 'ward')
    Z = sch.dendrogram(link, labels=decoder, orientation='left')
    axdendro.spines['bottom'].set_color('white')
    axdendro.spines['top'].set_color('white')
    axdendro.spines['left'].set_color('white')
    axdendro.spines['right'].set_color('white')
    axdendro.set_xticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.4, 0.1, 0.5, 0.8])
    index = Z['leaves']
    D = D[index, :]
    D = D[:, index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap='RdYlBu_r')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
    pylab.colorbar(im, cax=axcolor)
    if path:
        pylab.savefig(path, dpi=300, bbox_inches='tight')
    if show:
        pylab.show()
