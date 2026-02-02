import csv
import pickle
from pathlib import Path
from typing_extensions import Annotated
from typing import Tuple, List

import typer
from rich import print
import numpy as np
from tqdm import tqdm
import pandas as pd
import uproot


from . import GaussianProcessModel
from . import kernels
from . import _hist
from . import mass_resolution
from ._plot import plt, label


app = typer.Typer()

InputHistDefault = (Path('EventSelection_Data_10Percent.root'), 'h_Minv_General_Final_1')
InputHist = Annotated[
    Tuple[Path,str],
    typer.Option(help = 'Input histogram file with the key to the histogram in that file.')
]

the_kernel = (
    kernels.WhiteKernel(noise_level=7e3) +
    #kernels.RationalQuadratic()*kernels.ConstantKernel()
    #kernels.RationalQuadratic()*kernels.DotProduct()
    kernels.RBF(length_scale = 0.016)*kernels.DotProduct(sigma_0 = 2.5e4)
)
the_kernel_label = r'$K(m_i, m_j) = (\sigma_0^2 + m_i m_j \delta_{ij})e^{-(m_i-m_j)^2/\ell^2}$'


@app.command()
def fit_and_plot(
    name: Annotated[str, typer.Argument(help='name for this run, used as name for output files')],
    blind_range: Annotated[
        Tuple[float,float],
        typer.Option(
            help='edges of range to blind in GeV (exclusive with search-mass)',
        )
    ] = None,
    search_mass: Annotated[
        Tuple[float,float],
        typer.Option( help='search for this mass in GeV with the window size in standard deviations (exclusive with blind-range)')
    ] = None,
    empty_bin_variance: Annotated[
        float,
        typer.Option(
            help='use empty bins as values of 0 and include them with this variance'
            ' (otherwise, just drop empty bins)'
        )
    ] = None,
    low_lim : Annotated[float, typer.Option(help='lower limit of fit range in GeV')] = 0.033,
    up_lim : Annotated[float, typer.Option(help='upper limit of fit range in GeV')] = 0.179,
    rebin : Annotated[int,typer.Option(help='rebin factor to apply before fitting')] = 10,
    input : InputHist = InputHistDefault,
    out_dir : Annotated[Path, typer.Option(help='output directory to write files to')] = Path.cwd()
):
    """Fit a GP model and plot the result a single time"""
    if blind_range is not None and search_mass is not None:
        raise KeyError('You can only specify one of --blind-range or --search-mass.')

    br = None
    if blind_range is not None:
        br = blind_range
    elif search_mass is not None:
        center, width = search_mass
        width *= mass_resolution(center)
        br = [ center - width, center + width ]
    gpm = GaussianProcessModel(
        h = _hist.io.load(*input),
        kernel = the_kernel,
        blind_range = br,
        modify_histogram = [
            _hist.manipulation.rebin_and_limit(rebin, low_lim, up_lim),
        ],
        empty_bin_variance = empty_bin_variance
    )
    fig, axes = gpm.plot_comparison()
    filestem = out_dir / name
    fig.savefig(filestem.with_suffix('.png'), bbox_inches='tight')
    with open(filestem.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(gpm, f)


inject_signal = typer.Typer(
    help='inject signal into the IMD and write the resulting histogram to a file'
)

@inject_signal.command('single')
def inject_signal_single(
    output: Annotated[Path, typer.Argument(help='output file to write histogram to')],
    name: Annotated[str, typer.Argument(help='name of histogram in output file')],
    mass: Annotated[
        float,
        typer.Argument(
            help='inject signal of this mass in GeV',
            min = 0.35,
            max = 0.170
        )
    ],
    nevents : Annotated[int, typer.Argument(help='number of events to inject')],
    mass_width: Annotated[
        float,
        typer.Argument(
            help='width of signal peak',
            show_default='determined by mass resolution',
            min=0.0,
            max=0.02
        )
    ] = None,
    input : InputHist = InputHistDefault
):
    """Inject a known signal into the IMD"""
    h = _hist.io.load(*input)
    mass_width = mass_resolution(mass) if mass_width is None else mass_width
    h = _hist.manipulation.inject_signal(
        location = mass,
        width = mass_width,
        amplitude = nevents
    )(h)
    _hist.io.write(output, name, h)


@inject_signal.command('rand')
def inject_signal_rand(
    output: Annotated[Path, typer.Argument(help='output file to write histogram to, the `.root` suffix is used for the histograms and a `_answers.csv` lists the ID with the injected signal')],
    number: Annotated[int, typer.Option(help='number of signal bumps to inject, written as separate IMDs into the output file')] = 1,
    seed: Annotated[int, typer.Option(help='seed for random number generation')] = 1,
    nevents: Annotated[int, typer.Option(help='number of events to inject per mass')] = 10000,
    input : InputHist = InputHistDefault
):
    """Inject a random signal (or signals) into the IMD"""
    mass_selector = np.random.default_rng(seed = seed)
    h = _hist.io.load(*input)
    with (
        open(output.with_suffix('.csv'), 'w', newline='') as answers,
        uproot.recreate(output.with_suffix('.root')) as hist_f
    ):
        ans_f = csv.writer(answers)
        ans_f.writerow(['HID','Mass','Width','Ninjected'])
        for n in range(number):
            # cantor pairing for unique ID 
            #   ID is uniquely derived from seed and the number it was sampled from
            #   WARNING: if ID=42 is mass 0.080, then it will be mass 0.080 again
            hid = (n+seed)*(n+seed+1) // 2 + n
            mass = mass_selector.uniform(0.040,0.160)
            mass_width = mass_resolution(mass)
            # need to make a copy to avoid injecting two masses into the same histogram
            inj_h = h.copy()
            inj_h = _hist.manipulation.inject_signal(
                location = mass,
                width = mass_width,
                amplitude = nevents
            )(inj_h)
            hist_f[str(hid)] = inj_h
            ans_f.writerow([hid, mass, mass_width, nevents])


app.add_typer(inject_signal, name="inject-signal")

@app.command()
def search(
    output: Annotated[
        Path,
        typer.Argument(
            help='output directory to write results into'
        )
    ],
    mass_range: Annotated[
        List[float],
        typer.Option(
            help='mass range in GeV, '
            'specified like Python range (stop, start stop, start stop step) '
            'except the default for start is 0.040 and the default step is 0.005',
            )
    ] = [0.160],
    blind_halfwidth: Annotated[
        float,
        typer.Option(
            help='number of standard deviations defining blind window'
            ' (either side of mass)'
        )
    ] = 1.96,
    plot_each: Annotated[
        bool,
        typer.Option(
            help='Plot each fit into its own file.'
        )
    ] = False,
    input : InputHist = InputHistDefault
):
    """Search through mass points, performing a fit at each one"""
    start = 0.040
    stop  = None
    step  = 0.005
    if len(mass_range) > 3:
        raise ValueError('More than 3 parameters given as mass_range.')
    elif len(mass_range) < 1:
        raise ValueError('No values given to specify mass range.')
    elif len(mass_range) == 1:
        stop = mass_range[0]
    else:
        start = mass_range[0]
        stop  = mass_range[1]
        if len(mass_range) == 3:
            step = mass_range[2]
    output.mkdir(exist_ok=True, parents=True)
    mass_range = np.arange(start, stop, step)
    with open(output / 'search-results.csv', 'w', newline='') as f:
        o = csv.writer(f)
        o.writerow([
            'mass','sigma_m','chi2',
            'prediction', 'uncertainty', 'observation',
            # RBF*dotprod
            #'length_scale', 'sigma_0'
            # constant*RBF*dotprod
            #'constant_value', 'length_scale', 'sigma_0'
            # whitenoise + RBF * dotproduct
            'noise_level', 'length_scale','sigma_0'
        ])
        for mass, sigma_m in tqdm(zip(mass_range, mass_resolution(mass_range)), total=len(mass_range)):
            gpm = GaussianProcessModel(
                h = input,
                kernel = the_kernel,
                blind_range = (mass - blind_halfwidth*sigma_m, mass + blind_halfwidth*sigma_m),
                modify_histogram = [
                    _hist.manipulation.rebin_and_limit(10)
                ],
                empty_bin_variance = 3.688
            )
            if plot_each:
                out_name = output / f'{int(1000*mass)}mev_search'
                fig, axes = gpm.plot_comparison()
                fig.savefig(out_name.with_suffix('.png'), bbox_inches='tight')
                plt.close()
                with open(out_name.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(gpm, f)
    
            prediction, uncertainty = gpm.blind_single_bin_prediction
            observation = gpm.blind_single_bin_observation
            o.writerow([
                mass,
                sigma_m,
                np.sum(gpm.pull**2),
                prediction,
                uncertainty,
                observation,
                # getting kernel values when a constant is included
                #gpm.kernel.k1.k1.constant_value,
                #gpm.kernel.k1.k2.length_scale,
                #gpm.kernel.k2.sigma_0,
                # just DotProduct times RBF
                #gpm.kernel.k1.length_scale,
                #gpm.kernel.k2.sigma_0,
                # WhiteKernel + DotProduct * RBF
                gpm.kernel.k1.noise_level,
                gpm.kernel.k2.k1.length_scale,
                gpm.kernel.k2.k2.sigma_0
            ])

    s = pd.read_csv(output / 'search-results.csv')

    # TODO: figure out how to extract when the fit knows it failed
    # so we don't have to do this proxy check
    good = (s.chi2 > 200)

    fig, axes = plt.subplots(
        nrows = 3,
        sharex = 'col',
        gridspec_kw = dict(
            hspace = 0.05
        )
    )

    axes[0].annotate(
        the_kernel_label,
        (0.5, 0.9),
        xycoords = 'axes fraction',
        va = 'top',
        ha = 'center'
    )

    for ax, column, ylabel in zip(
        axes,
        ['chi2', 'length_scale', 'sigma_0'],
        [r'$\chi^2$', r'$\ell$ / GeV', r'$\sigma_0$ / GeV']
    ):
        ax.plot(s[good].mass, s[good][column])
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel('mass / GeV')
    label(ax = axes[0])
    fig.savefig(output / 'search-good-fit-overview.png', bbox_inches='tight')
    plt.close() 


if __name__ == '__main__':
    app()
