# Copyright (c) 2014, Mayo Foundation For Medical Education and Research
# All rights reserved.
#
# Written by Rob Reid. 

import numpy as np
    
def condition_seeds(seeds, aff, boxshape, tol=0.2, fudgefac=0.05, verbose=3,
                    printprecision=2):
    """
    dipy.tracking.eudx will abort if any of the seeds are outside of the
    peaks.peak_values array.  This checks for seeds outside the box and 
    * nudges them inside if they are within tol of it (tiny excursions due to
      numerical imprecision in the affine calculations are common), or
    * removes them if necessary

    Parameters
    ----------
    seeds: np.ndarray
        (N, 3) list of seeds in world coords.
    aff: np.ndarray
        Affine matrix.  world coords = np.dot(aff, voxel coords)
    boxshape: tuple
        e.g. peaks.peak_values.shape[:3]
    tol: float
        The tolerance in voxel coords
    fudgefac: float
        Nudges will add an extra fudgefac * tol to avoid having seeds on the edge get reflagged.
    verbose: int
        0: Be vewy vewy quiet
        1: Provide a summary of changes.
        2: Also report each removal.
        3: Report on any adjustment.
    printprecision: int
        Number of digits to print after each decimal point.

    Output
    ------
    goodseeds: np.ndarray
        (M, 3) list of good seeds in world coords.
    """
    affinv = np.linalg.inv(aff)
    maxs = np.array(boxshape) - 1.0
    fudge = fudgefac * tol
    fudgedmaxs = maxs - fudge
    tol2 = tol**2
    goodones = []
    nrmed = 0
    nnudged = 0
    printdefaults = np.get_printoptions()
    np.set_printoptions(precision=2)
    ijk1 = np.ones(4)
    for s in seeds:
        vox = np.dot(affinv, (s[0], s[1], s[2], 1))[:3]
        if np.any(vox < 0.0) or np.any(vox > maxs):
            clipped = np.clip(vox, (0.0, 0.0, 0.0), maxs)
            dist2 = np.sum((vox - clipped)**2)
            if dist2 > tol2:
                nrmed += 1
                if verbose > 1:
                    print "Removed %s (voxel coords %s)" % (s, vox)
            else:
                ijk1[:3] = np.clip(vox, fudge, fudgedmaxs)
                goodones.append(np.dot(aff, ijk1)[:3])
                nnudged += 1
                if verbose > 2:
                    print "Nudged %s (voxel coords %s)" % (s, vox)
        else:
            goodones.append(s)
    if verbose > 0:
        print "Removed %d seeds" % nrmed
        print "Nudged %d seeds" % nnudged
    np.set_printoptions(**printdefaults)
    return np.array(goodones)
