from __future__ import annotations

from pathlib import Path

import numpy as np

from locate_point3D import locate_point3D
from matlab_compat import load_numeric_excel, xlswrite_compat


def obs_vs_poseidon(*args):
    nargin = len(args)
    if nargin == 0:
        ofile = '1986.5.dat'
    else:
        ofile = args[0]
    if nargin < 2:
        obs_folder = '../../measurements'
    else:
        obs_folder = args[1]
    if nargin < 3:
        fname_boxes = 'Box_surfareas.xlsx'
    else:
        fname_boxes = args[2]
    if nargin < 4:
        bdepths = [25, 100, 500, 2000]
    else:
        bdepths = args[3]
    if nargin < 5:
        obs_pos_outfile = None
    else:
        obs_pos_outfile = args[4]

    BBB = load_numeric_excel(fname_boxes)
    boxnum = BBB[:, 0]
    boxlats = np.zeros((BBB.shape[0], 2), dtype=float)
    boxlats[:, 0] = BBB[:, 4]
    boxlats[:, 1] = BBB[:, 5]
    boxlons = np.zeros((BBB.shape[0], 2), dtype=float)
    boxlons[:, 0] = BBB[:, 6]
    boxlons[:, 1] = BBB[:, 7]
    boxdepths = BBB[:, 1]
    boxnlev = BBB[:, 8].astype(int)
    Nbox = boxnum.shape[0]
    Maxlev = int(np.max(boxnlev))

    ofile = str(Path(obs_folder) / ofile)
    data = np.loadtxt(ofile, skiprows=1)
    data = np.atleast_2d(data)
    slon = data[:, 1]
    slat = data[:, 2]
    sval = data[:, 3]
    Ns = slon.shape[0]
    sdepth = np.zeros((Ns, 1), dtype=float)
    sdepth[:, 0] = 10

    nobs_box = np.zeros((Nbox, Maxlev), dtype=int)
    box_st = {(j, k): {'ind': [], 'val': []} for j in range(Nbox) for k in range(Maxlev)}
    box_vs_allobs = np.zeros((Ns, 4), dtype=float)

    for i in range(Ns):
        bind, bn, blev = locate_point3D(slon[i], slat[i], sdepth[i, 0], boxnum, boxlons, boxlats, boxdepths, boxnlev, bdepths)
        bind = np.asarray(bind).ravel()
        bn = np.asarray(bn).ravel()
        blev = int(np.asarray(blev).ravel()[0])

        if 1 <= bind.size <= 4:
            for j in range(bind.size):
                bj = int(bind[j])
                if 0 <= bj < Nbox:
                    bind0 = bj
                elif 1 <= bj <= Nbox:
                    bind0 = bj - 1
                else:
                    raise IndexError(f'bind out of range: {bj}')
                box_vs_allobs[i, j] = bn[j]
                nobs_box[bind0, blev - 1] += 1
                box_st[(bind0, blev - 1)]['ind'].append(i)
                box_st[(bind0, blev - 1)]['val'].append(sval[i])
        elif bind.size > 4:
            print('Error from obs_boseidon: obs belongs to more than 4 boxes: i=')
            print(i + 1)
            raise ValueError('obs belongs to more than 4 boxes')

    box_vs_allobs = np.concatenate((slon.reshape(-1, 1), slat.reshape(-1, 1), sval.reshape(-1, 1), box_vs_allobs), axis=1)

    obs = []
    brefnum = []
    breflev = []
    sigc = []
    for j in range(Nbox):
        for k in range(Maxlev):
            if nobs_box[j, k] == 0:
                continue
            brefnum.append(boxnum[j])
            breflev.append(k + 1)

            nn = nobs_box[j, k]
            obsL = 0.0
            for ii in range(nn):
                idx = box_st[(j, k)]['ind'][ii]
                obsL += sval[idx]
            obsL = obsL / np.single(nn)
            obs.append(obsL)
            if nn == 1:
                sigc.append(0.0)
            else:
                vals = np.asarray(box_st[(j, k)]['val'], dtype=float)
                sigc.append(0.5 * (np.max(vals) - np.min(vals)))

    obs = np.asarray(obs, dtype=float).reshape(-1, 1)
    brefnum = np.asarray(brefnum).reshape(-1, 1)
    breflev = np.asarray(breflev).reshape(-1, 1)
    sigc = np.asarray(sigc, dtype=float).reshape(-1, 1)

    if obs_pos_outfile is not None:
        xlswrite_compat(str(obs_pos_outfile), obs, 'obs')
        xlswrite_compat(str(obs_pos_outfile), sigc, 'sigc')
        xlswrite_compat(str(obs_pos_outfile), brefnum, 'bnum')
        xlswrite_compat(str(obs_pos_outfile), box_vs_allobs, 'box_vs_allobs')

    return obs, brefnum, breflev, box_vs_allobs, boxlats, boxlons
