class Element:
    H, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, P = list(range(1, 16))
    S, Cl, Ar, K, Ca, Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn = list(range(16, 31))
    Ga, Ge, As, Se, Br, Kr, Rb, Sr, Y, Zr, Nb, Mo, Tc, Ru = list(range(31, 45))
    Rh, Pd, Ag, Cd, In, Sn, Sb, Te, I, Xe, Cs, Ba, La, Ce = list(range(45, 59))
    Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Hf = list(range(59, 73))
    Ta, W, Re, Os, Ir, Pt, Au, Hg, Tl, Pb, Bi, Po, At, Rn = list(range(73, 87))
    Fr, Ra, Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm = list(range(87, 101))
    Md, No, Lr, Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh = list(range(101, 114))
    Fl, Mc, Lv, Ts, Og = list(range(114, 119))

    n_elements = 119


ELEMENT_DICT = {getattr(Element, attr): attr for attr in dir(Element) if
                not attr.startswith('_') and attr != 'n_elements'}

monoisotopic = [
    'Na', 'Be', 'F', 'Al', 'P', 'Sc', 'V', 'Mn', 'Co', 'As', 'Rb', 'Y', 'Nb', 'Tc', 'Rh', 'In', 'I', 'Cs', 'Re',
    'Au', 'Bi', 'La', 'Pr', 'Eu', 'Tb', 'Lu', 'Ho', 'Tm', 'Pa'
]


def lorentzian(x, x0, gam):
    return gam ** 2 / (gam ** 2 + (x - x0) ** 2)
