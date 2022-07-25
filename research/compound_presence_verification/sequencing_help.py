import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath('..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from pyteomics.mass import calculate_mass

from mass_automation.experiment import Experiment, Spectrum
from mass_automation.formula import Formula
from mass_automation.formula.check_formula import check_presence
from mass_automation.formula.plot import plot_compare


def gen_molec(chain, ion_type):
    aminoacid_dict = {
    'A':   Formula({'H': 5, 'C': 3, 'O': 1, 'N': 1}),
    'C':   Formula({'H': 5, 'C': 3, 'S': 1, 'O': 1, 'N': 1}),
    'D':   Formula({'H': 5, 'C': 4, 'O': 3, 'N': 1}),
    'E':   Formula({'H': 7, 'C': 5, 'O': 3, 'N': 1}),
    'F':   Formula({'H': 9, 'C': 9, 'O': 1, 'N': 1}),
    'G':   Formula({'H': 3, 'C': 2, 'O': 1, 'N': 1}),
    'H':   Formula({'H': 7, 'C': 6, 'N': 3, 'O': 1}),
    'I':   Formula({'H': 11, 'C': 6, 'O': 1, 'N': 1}),
    'J':   Formula({'H': 11, 'C': 6, 'O': 1, 'N': 1}),
    'K':   Formula({'H': 12, 'C': 6, 'N': 2, 'O': 1}),
    'L':   Formula({'H': 11, 'C': 6, 'O': 1, 'N': 1}),
    'M':   Formula({'H': 9, 'C': 5, 'S': 1, 'O': 1, 'N': 1}),
    'N':   Formula({'H': 6, 'C': 4, 'O': 2, 'N': 2}),
    'P':   Formula({'H': 7, 'C': 5, 'O': 1, 'N': 1}),
    'Q':   Formula({'H': 8, 'C': 5, 'O': 2, 'N': 2}),
    'R':   Formula({'H': 12, 'C': 6, 'N': 4, 'O': 1}),
    'S':   Formula({'H': 5, 'C': 3, 'O': 2, 'N': 1}),
    'T':   Formula({'H': 7, 'C': 4, 'O': 2, 'N': 1}),
    'V':   Formula({'H': 9, 'C': 5, 'O': 1, 'N': 1}),
    'W':   Formula({'C': 11, 'H': 10, 'N': 2, 'O': 1}),
    'Y':   Formula({'H': 9, 'C': 9, 'O': 2, 'N': 1}),
    'U':   Formula({'H': 5, 'C': 3, 'O': 1, 'N': 1, 'Se' : 1}),
    'O':   Formula({'H': 19, 'C': 12, 'O': 2, 'N': 3})
    }
    
    aa_list = list(chain)
    formula = Formula('')
    
    for aa in aa_list:
        formula = formula + aminoacid_dict[aa]
    
    if ion_type == 'a':
        formula += Formula('H')
        formula -= Formula('CO')
        
    elif ion_type == 'b':
        formula += Formula('H')
        
    elif ion_type == 'c':
        formula += Formula('NH4')
        
    elif ion_type == 'x':
        formula += Formula('OH')
        formula += Formula('CO')
    
    elif ion_type == 'y':
        formula += Formula('OH3')
        
    elif ion_type == 'z':
        formula += Formula('OH')
        formula -= Formula('NH')
    else:
        raise ValueError('Unknown ion type')
        
    return formula


def fragments(peptide, types=('b', 'y')):
    """ The function generates all possible m/z for fragments of types and creates dictionary where keys are tuples with aminoacid
    chain and ion type
    """
    fragments = {}
    for ion_type in types:
        for i in range(1, len(peptide)+1):
            if ion_type in 'abc':
                fragments[(peptide[:i], ion_type)] = gen_molec(peptide[:i], ion_type)  
            else:
                fragments[(peptide[i-1:], ion_type)] = gen_molec(peptide[i-1:], ion_type)
                
    return fragments


def plot_seq(spectrum: Spectrum, peptide: str, types: tuple, threshold=0.052, path=None):
    """ Function provides de novo sequencing and plots results
    """
    mpl.rcParams['font.family'] = 'Arial'
    hfont = {'fontname':'Arial', 'fontsize':'25', 'weight': 'bold'}
    tfont = {'fontname':'Arial', 'fontsize':'30'}
    sfont = {'fontname': 'Arial', 'fontsize': '12'}
    fragments_ = fragments(peptide, types=types)
    
    columns = {}
    for i, elem in enumerate(types):
        columns[elem] = i

    rows = [i+1 for i in range(len(peptide))]
    cell_text = [['Not found'] * len(types) for i in range(len(peptide))]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 2]}, figsize=(30, 10))
    max_int = spectrum.ints.max()
    
    masses = []
    ints = []
    
    for key in tqdm(fragments_.keys()):
        chain = key[0]
        ion_type = key[1]
        number = len(chain)
        res = check_presence(spectrum, fragments_[key], cal_error=0.006, max_peaks=1)
        cosine_dist = res[0]
        if cosine_dist < threshold:
            real_masses, real_ints = res[1]
            mass = real_masses[np.argmax(real_ints)]
            cell_text[number-1][columns[ion_type]] = round(mass, 4)
            int_ = real_ints.max()
            ax1.annotate(f'${ion_type}_{{{number}}}$', (mass, int_+ max_int*0.05), ha='center', **tfont)
            masses.append(mass)
            ints.append(int_)
            
     
    ax1.plot(spectrum.masses, spectrum.ints)
    ax1.spines["left"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.vlines(np.array(masses), 0, np.array(ints)+max_int*0.01, linewidth=4, colors=("orange"))  
    ax1.set_title(f"Sequencing of {peptide}", **hfont)
    ax1.set_xlim(160, 1300)
    ax1.set_ylim(0, 1.2*max_int)
    ax1.set_xlabel("m/z", **sfont)
    ax1.set_ylabel("Intensity", **sfont)
    

    the_table = ax2.table(cell_text, 
          rowLabels=rows,
          colLabels=types,
          loc='upper center',
          rowColours =["orange"] * len(peptide),
          colColours =["orange"] * len(types)
          )
    
    ax2.axis('off')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()
         