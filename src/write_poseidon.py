from __future__ import annotations

from pathlib import Path


def write_poseidon(fname, vect, Lind, bnlev, header, footer, sed):
    vect = list(vect)
    path = Path(fname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fid:
        for item in header:
            fid.write(f"{item['tline']}\n")
        Nbox = len(bnlev) - 1
        for i in range(1, Nbox + 1):
            fid.write(f"   {i:2d}")
            for j in range(1, int(bnlev[i]) + 1):
                fid.write(f"   {vect[Lind[i - 1, j - 1] - 1]:9.3E}")
            fid.write(' ' * ((4 - int(bnlev[i])) * 12))
            fid.write(f"   {sed[i - 1, 0]:9.3E}   {sed[i - 1, 1]:9.3E} \n")
        for item in footer:
            fid.write(f"{item['tline']}\n")
