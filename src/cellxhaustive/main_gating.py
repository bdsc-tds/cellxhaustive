"""
AT. Add general description here.
"""


# Import modules
import copy
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture


# Perform main gating strategy to separate CD4T, CD8T, Monocytes, DCs, NKs, and B cells
def gaussian_gating(mat, markers, marker_order=['CD3', 'CD4'],
                    positive=[True, True], makeplot=False, random_state=None,
                    hao_extension='.2', root='./'):
    """
    Main gating strategy

    Parameters:
    -----------
    mat: array(float)
      2-D numpy array expression matrix, with cells in D0 and markers in D1.
      In other words, rows contain cells and columns contain markers.

    markers: array(str)
      1-D numpy array with markers matching each column of mat.



    marker_order: list(str) (default=['CD3', 'CD4'])  # AT. Check if it really is default or best to leave empty?
      List of markers used in the gating strategy ordered accordingly.

    positive: list(bool) (default=[True, True])  # AT. Likewise, check if it really is default or best to leave empty?
      List describing wether the markers in marker_order are positively
      (True) or negatively (False) expressed.

    hao_extension: str (default='.2')
       The hao dataset uses multiple clones for some of the antibodies, e.g.
       naming them as CD4.1, CD4.2, etc. This variable allows you to add the
       extension for the right clone. There are certainly better and more
       generalizable ways to do so, but I just can't be bothered...
       # AT. Double check this and update description



    makeplot: bool (default=False)
      Whether or not to generate a figure.

    random_state: int or None (default=42)
       Random seed.



    root: str (default='./')
      Output path of the figure.
      # AT. Improved parameter description. Maybe reorder to put it next to makeplot

    Returns:
    --------
      # AT. Add what is returned by the function
    """

    # Use two vectors to filter progressively
    truefalse = np.zeros(np.shape(mat)[0]) == 0
    truefalse_ = np.zeros(np.shape(mat)[0]) != 0

    for idx, marker in enumerate(marker_order):
        # Find expression for a given marker index
        mdx = np.where(markers == marker)[0]
        if len(mdx) == 0:
            mdx = np.where(markers == marker + hao_extension)[0][0]
        else:
            mdx = mdx[0]

        # Select expression for such marker
        expression = mat[truefalse, mdx]

        # Figure out what the peaks are along the x axis.
        xran = np.linspace(np.min(expression), np.max(expression), num=10000)
        gm = GaussianMixture(n_components=2, random_state=random_state,
                             max_iter=100, n_init=10,  # AT. Double-check this 'tol=1e-5,'
                             means_init=np.array([[1], [5]])).fit(expression[:, np.newaxis])

        # Sort peaks
        means_ = gm.means_[:, 0]
        order = np.argsort(means_)
        means_ = means_[order]

        # Find density kde for the peaks and figure out where the valley is.
        kernel = stats.gaussian_kde(expression)
        density = kernel.evaluate(xran)
        xran_ = np.logical_and(xran > means_[0], xran < means_[1])
        density_ = density[xran_]
        xran_ = xran[xran_]
        kdx = np.argmin(density_)
        valley = xran_[kdx]

        # Based on the position of the valley, select the positive or negative cells for marker idx
        if positive[idx]:
            truefalse_[truefalse] = expression > valley
            truefalse = copy.copy(truefalse_)
            truefalse_ = truefalse_ * False
        else:
            truefalse_[truefalse] = expression < valley
            truefalse = copy.copy(truefalse_)
            truefalse_ = truefalse_ * False

    return truefalse


def main_gating():

    return True
# AT. Check if useful
