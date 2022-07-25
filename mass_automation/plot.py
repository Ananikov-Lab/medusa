from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge

from .experiment import Spectrum
from .utils import monoisotopic


def plot_spectrum(spectrum: Spectrum, drawtype='plot', x_left=None, x_right=None, y_max=None,
                  labels=None, rel=False, path=None, figsize=(9, 6), show=True, annotate_distributions=False):
    """Performs spectrum plotting and visualization.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum you want to plot.
    drawtype : str
        The visualization method. ``'plot'`` is equivalent to ``plt.plot``, ``'scatter'`` is equivalent to ``plt.scatter``,
        ``'vlines'`` is equivalent to ``plt.vlines``.(default is ``'plot'``)
    x_left : float
        The left limit of the graph.
    x_right : float
        The right limit of the graph.
    y_max : float
        The upper limit of the graph. If ``y_max`` is -1, the upper limit is the local maximum
        of [``x_left``, ``x_right``]. If ``y_max`` is None, the upper limit is the global maximum of the spectrum.
    labels : np.ndarray
        The array of labels. Can be a result of Deisotoper, find_peaks or self marked data. Must have the same length
        as the spectrum arrays. Non shown indices must be labeled as -1. Labels should be integers from 0 to maximal
        label.
    rel: bool
        If True, creates a graph with relative intensities.
    path: str
        The path where the graph should be saved.
    figsize : tuple
        The tuple where the first and the second numbers are the figure's width and height respectively.
    show : bool
        If True, show the graph, else not
    annotate_distributions : bool
        If True, annotate the distributions of the peaks.

    Raises
    -------
    ValueError
        Incorrect drawing type.
    """
    masses = spectrum.masses
    ints = spectrum.ints

    if rel:
        ints = ints / ints.max()
        ylabel = 'Rel. intensity'
    else:
        ylabel = 'Intensity'

    if x_left is None:
        x_left = masses[0]
    if x_right is None:
        x_right = masses[-1]
    slice_masses = masses[(masses >= x_left) & (masses <= x_right)]
    slice_ints = ints[(masses >= x_left) & (masses <= x_right)]
    if y_max is None:
        y_max = ints.max()
    elif y_max == -1:
        y_max = slice_ints.max()

    plt.figure(figsize=figsize)
    plt.xlabel("m/z")
    plt.ylabel(ylabel)
    if drawtype == 'plot':
        plt.plot(slice_masses, slice_ints)
    elif drawtype == 'scatter':
        plt.scatter(slice_masses, slice_ints)
    elif drawtype == 'vlines':
        plt.vlines(slice_masses, 0, slice_ints)
    else:
        raise ValueError('Incorrect drawing type')

    plt.ylim(0, y_max + 0.1 * y_max)

    if labels is not None:
        slice_labels = labels[(masses >= x_left) & (masses <= x_right)]
        counter = len(np.unique(labels)) - 1
        randomized_colors = np.arange(counter)
        np.random.shuffle(randomized_colors)
        plt.scatter(slice_masses[slice_labels != -1], slice_ints[slice_labels != -1], marker="o",
                    s=20, c=randomized_colors[slice_labels.astype(int)][slice_labels != -1], cmap='jet')

        if annotate_distributions:
            for i in range(int(labels.max() + 1)):
                plt.annotate(
                    str(i),
                    [masses[labels == i].mean(), ints[labels == i].max()]
                )

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def plot_periodic_table(values: dict, file_path: str, normalize: Optional[bool] = True,
                        show_mono: Optional[bool] = False):
    """Plots periodic table heatmap

    Parameters
    ----------
    values : dict
        List of values, should contain values for all
    file_path : str
        Path to the output html file
    normalize : bool
        Whether normalize spectrum or not
    show_mono : bool
        Highlight mono isotopic elements
    """

    periods = ["I", "II", "III", "IV", "V", "VI", "VII"]
    groups = [str(x) for x in range(1, 19)]

    df = elements.copy()
    df["atomic mass"] = df["atomic mass"].astype(str)
    df["group"] = df["group"].astype(str)
    df["period"] = [periods[x - 1] for x in df.period]

    count = 0
    for i in range(56, 71):
        df.period[i] = 'La'
        df.group[i] = str(count + 3)
        count += 1

    count = 0
    for i in range(88, 103):
        df.period[i] = 'Ac'
        df.group[i] = str(count + 3)
        count += 1

    df['values'] = df['atomic number'].map(values)
    df['values'][df['values'] < 0] = 0

    if normalize:
        df['values'] /= df['values'].max()
    df['values'] = df['values'].map(lambda x: to_hex(plt.cm.RdYlGn(x) if not pd.isna(x) else plt.cm.Reds(.1)))

    df['is_monoisotopic'] = df['symbol'].map(lambda x: x in monoisotopic)

    output_file(file_path)

    p = figure(plot_width=800, plot_height=500,
               x_range=groups, y_range=["Ac", "La", ""] + list(reversed(periods)),
               toolbar_location=None)
    p.output_backend = "svg"

    r = p.rect("group", "period", 0.95, 0.95, source=df, fill_alpha=0.6,
               color='values')

    if show_mono:
        r = p.rect("group", "period", 0.95, 0.95, source=df[df['is_monoisotopic']], fill_alpha=0,
                   line_color='black')

    text_props = {"source": df, "text_align": "left", "text_baseline": "middle"}

    x = dodge("group", -0.4, range=p.x_range)

    p.text(x=x, y="period", text="symbol", text_font_style="bold", **text_props)

    p.text(x=x, y=dodge("period", 0.3, range=p.y_range), text="atomic number",
           text_font_size="8pt", **text_props)

    p.text(x=["3", "3"], y=["VI", "VII"], text=["LA", "AC"], text_align="center", text_baseline="middle")
    p.axis.visible = False
    p.outline_line_color = None
    p.grid.grid_line_color = None

    show(p)
