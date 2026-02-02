import pickle
from pathlib import Path
from typing import Tuple, Union

import scipy
import uproot
import hist

def sim_imd():
    """simulate the IMD by sampling from a moyal distribution

    The values of the two moyal distribution parameters were taken
    from a fit of a Moyal PDF to the 6.5% 2016 IMD. The returned
    histogram has the same binning as this IMD and roughly the same
    number of entries (~5M, some lost to overflow).

    Returns
    -------
    hist.Hist
        simulated IMD
    """
    return (
        hist.Hist.new
        .Reg(6000, 0.0, 0.3, label = 'Mass / GeV')
        .Double()
        .fill(
            scipy.stats.moyal.rvs(
                loc = 0.065,
                scale = 0.013,
                size = 5_000_000
            )
        )
    )


def load(fp: Union[str, Path], name: str = 'invM_h'):
    """Load an IMD from the input file and make sure the x axis is labeled appropriately

    Parameters
    ----------
    filepath: str|Path
        path to ROOT file to load
    imd_name: str, optional, default invM_h
        key name of histogram in ROOT file

    Returns
    -------
    hist.Hist
        loaded IMD from the input file
    """
    with uproot.open(fp) as f:
        h = f[name].to_hist()
        h.axes[0].label = 'Mass / GeV'
        return h


def write(fp: Union[str, Path], name: str, h: hist.Hist):
    """Write a hist.Hist object to the output file under the given name

    Parameters
    ----------
    fp: str|Path
        path to file to write to
        extension of this file determines what file type will be written
    name: str
        name to write histogram under within the file
    h: hist.Hist
        histogram whose contents should be written
    """

    fp = Path(fp)
    if fp.suffix == '.root':
        with uproot.recreate(fp) as f:
            f[name] = h
    elif fp.suffix == '.pkl':
        with open(fp, 'wb') as f:
            pickle.dump({name: h}, f)
    else:
        raise ValueError(r"Extension '{fp.suffix}' not recognized as pickle ('.pkl') or ROOT ('.root')")


def _deduce_histogram(h: Union[hist.Hist, str, Tuple[str, str]]):
    """Deduce and return the histogram that should be used from the input specification

    Meant to be used within the construction of the GP model class below.

    Parameters
    ----------
    h: hist.Hist|str
        If a hist.Hist is given, use that as the histogram.
        If h is a str, there are two possible values.
        'sim' returns the result of sim_imd and 'real' returns the result of 'load_imd'
        with the filepath 'hps2016invMHisto10pc.root'.
        Any other str will be treated as a filepath that contains a histogram
        named invM_h.

    Returns
    -------
    hist.Hist
        histogram following input specification
    """

    if isinstance(h, str):
        if h == 'sim':
            return sim_imd()
        elif h == 'real':
            return load('hps2016invMHisto10pc.root')
        else:
            return load(h)
    elif isinstance(h, tuple):
        return load(h[0], h[1])
    elif isinstance(h, hist.Hist):
        return h
    else:
        raise TypeError(f'Histogram specification of type {type(h)} not supported.')
