import numpy as np
import matplotlib.pyplot as plt

from . import Formula
from ..experiment import Spectrum
from mass_automation.formula.check_formula import del_isotopologues, check_presence


def plot_compare(spectrum: Spectrum, formula: Formula, cal_error=0.006,
                 dist_error=0.003, distance=50, max_peaks=5, path=None,
                 return_results=False, max_int='all', return_masses=False, figsize=(10, 7),
                 show_text=True, off_axis=False, show=True):
    """ Spectra comparison visualization.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum, where algorithm tries to detect the substance.
    formula : Formula
        The Formula class of the substance.
    cal_error : float
        The radius of the first peak' vicinity (default is 0.006)
    dist_error : float
        The possible error in distance between the peaks, characterizes the radius of the second and the next
        peak's vicinity (default is 0.001).
    distance : int
        The parameter, used in peak finding algorithm. (default is 50)
    max_peaks : int
        Limits the number of possible first peaks (default is 5)
    path : str
        Saves plotting figure in png with dpi = 300.
    return_results : bool
        If True, check presense output is returned (default is False)
    return_masses : bool
        If True, detected m/z values are returned. It works only, if return_results is set as False (default is False)
    figsize : tuple
        Determines size of the plot (default is (10, 7))
    show_text: bool    
        If True, show text description on plot
    off_axis: bool
        If True, set axis invisible
    show : bool
        If True, show plot else not (default is True)

    """
    available_fonts = [f.name for f in fm.fontManager.ttflist]
   
    if show_text:
        hfont_size = '12'
    else:
        hfont_size = '18'
    sfont_size = '9'
    
    if 'Arial' in available_fonts:
        
        hfont = {'fontname': 'Arial', 'fontsize': hfont_size}
        sfont = {'fontname': 'Arial', 'fontsize': sfont_size}
        
    elif 'Nimbus Sans' in available_fonts:
        
        hfont = {'fontname':'Nimbus Sans', 'fontsize': hfont_size}
        sfont = {'fontname': 'Nimbus Sans', 'fontsize': sfont_size}
        
    else:
        
        hfont = {'fontname':'DejaVu Sans', 'fontsize': hfont_size}
        sfont = {'fontname': 'DejaVu Sans', 'fontsize': sfont_size}
        
    teor_masses, teor_ints = formula.isodistribution()
    teor_masses, teor_ints = del_isotopologues(teor_masses, teor_ints)
    #import ipdb; ipdb.set_trace()
    cosine_distance, real_coords, mtchd_p_per, mass_delta = check_presence(spectrum,
                                                                           formula,
                                                                           cal_error,
                                                                           dist_error,
                                                                           distance,
                                                                           max_peaks)
    real_masses, real_ints = real_coords
    plt.figure(figsize=figsize)
    x_left = int(round(teor_masses[0] - 1, 0))
    x_right = int(round(teor_masses[-1] + 1, 0))
    xticks = np.linspace(x_left, x_right, x_right - x_left + 1)
    ax1 = plt.subplot(212)
    
    ax1.vlines(teor_masses, 0, teor_ints)
    if show_text:
        for i in range(len(teor_masses)):
            ax1.annotate(str(round(teor_masses[i], 4)),
                         (teor_masses[i], teor_ints[i]+0.04), ha='center', **sfont)
            
    ax1.set_xlim(x_left, x_right)
    ax1.plot([x_left, x_right], [0, 0])
    ax1.set_ylim(0, 1.1)
    ax1.set_xticks(xticks)
    
    ax1.annotate("Calculated ",
                 (x_right, teor_ints.max()), weight='bold', ha='right', **hfont)
    
    if off_axis:
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.axes.get_yaxis().set_visible(False)
        

    ax2 = plt.subplot(211)
    ax2.plot(spectrum.masses, spectrum.ints)
    ax2.scatter(real_masses, real_ints, c="orange")
    if show_text:
        for i in range(len(real_masses)):
            ax2.annotate(str(round(real_masses[i], 4)),
                         (real_masses[i], real_ints[i]+0.04*real_ints.max()),
                         ha='center', **sfont)
            
    ax2.set_xlim(x_left, x_right)
    plot_slice_condition = (spectrum.masses >= x_left) & (spectrum.masses <= x_right)
    if max_int == 'all':
        plot_slice_max = spectrum.ints[plot_slice_condition].max()
    elif max_int == 'searched':
        plot_slice_max = real_ints[0]
    else:
        raise ValueError('Max_int parameter value is not correct')
        
    ax2.set_ylim(0, 1.1 * plot_slice_max)
    ax2.set_xticks(xticks)
    ax1.set_xlabel("m/z", **hfont)
    
    ax2.annotate("Experimental ", (x_right, plot_slice_max), ha='right', weight='bold', **hfont)
    
    if off_axis:
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.axes.get_yaxis().set_visible(False)
    
    
    if show_text:
        ax2.annotate(r"$\Delta$" + f" = {round(mass_delta, 2)} ppm ",
                     (x_right, 0.5*plot_slice_max), ha='right', **hfont)
        ax2.annotate("Cos. dist. = {:.1e} ".format(cosine_distance),
                     (x_right, 0.57*plot_slice_max), ha='right', **hfont)
        ax2.annotate(f"Matched peaks percentage = {round(mtchd_p_per*100, 1)} %",
                     (x_right, 0.64*plot_slice_max), ha='right', **hfont)

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()
    
    if return_results:
        return cosine_distance, real_coords, mtchd_p_per, mass_delta
    
    if return_masses:
        return real_masses
