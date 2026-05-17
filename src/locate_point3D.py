from __future__ import annotations


def locate_point3D(slon, slat, sdepth, boxnum, boxlons, boxlats, boxdepths, boxnlev, bdepths):
    bind = []
    bn = []
    for i in range(len(boxnum)):
        if slon <= boxlons[i, 1] and slon >= boxlons[i, 0] and slat <= boxlats[i, 1] and slat >= boxlats[i, 0]:
            bind.append(i)
            bn.append(int(boxnum[i]))
    if not bind:
        raise ValueError(f'{[slon, slat]} is outside box system, slon')

    blev = 0
    for j in range(int(boxnlev[bind[0]])):
        if sdepth < bdepths[j]:
            blev = j + 1
            break
    if blev == 0:
        blev = int(boxnlev[bind[0]])
    return bind, bn, blev
