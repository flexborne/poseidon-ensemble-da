from __future__ import annotations

import numpy as np


def Box_Neighbors(Nbox, brefnum=None, boxlats=None, boxlons=None):
    BoxNb = np.eye(Nbox, dtype=int)

    BoxNb[3, 4:8] = 1
    BoxNb[4, 6:11] = 1
    BoxNb[5, 6] = 1; BoxNb[5, 31:33] = 1
    BoxNb[6, 7] = 1; BoxNb[6, 30:33] = 1
    BoxNb[7, 8] = 1; BoxNb[7, 29:32] = 1
    BoxNb[8, 9] = 1; BoxNb[8, 28:31] = 1
    BoxNb[9, 10] = 1; BoxNb[9, 27:30] = 1
    BoxNb[10, 11] = 1; BoxNb[10, 26:29] = 1
    BoxNb[11, 12] = 1; BoxNb[11, 25:28] = 1
    BoxNb[12, 13:15] = 1; BoxNb[12, 24:27] = 1
    BoxNb[13, 14:17] = 1
    BoxNb[14, 15:17] = 1; BoxNb[14, 23:26] = 1
    BoxNb[15, 16:18] = 1; BoxNb[15, 22:25] = 1
    BoxNb[16, 17:20] = 1
    BoxNb[17, 18:20] = 1; BoxNb[17, 21:24] = 1
    BoxNb[18, 19:23] = 1
    BoxNb[19, 20] = 1
    BoxNb[20, 21] = 1
    BoxNb[21, 22] = 1; BoxNb[21, 42] = 1
    BoxNb[22, 23] = 1; BoxNb[22, 42] = 1
    BoxNb[23, 24] = 1; BoxNb[23, 41:43] = 1
    BoxNb[24, 25] = 1; BoxNb[24, 40:43] = 1
    BoxNb[25, 26] = 1; BoxNb[25, 39:42] = 1
    BoxNb[26, 27] = 1; BoxNb[26, 38:41] = 1
    BoxNb[27, 28] = 1; BoxNb[27, 37:40] = 1
    BoxNb[28, 29] = 1; BoxNb[28, 36:39] = 1
    BoxNb[29, 30] = 1; BoxNb[29, 35:38] = 1
    BoxNb[30, 31] = 1; BoxNb[30, 34:37] = 1
    BoxNb[31, 32] = 1; BoxNb[31, 33:36] = 1
    BoxNb[32, 33:35] = 1
    BoxNb[33, 34] = 1
    BoxNb[34, 35] = 1; BoxNb[34, 43:45] = 1
    BoxNb[35, 36] = 1; BoxNb[35, 43:46] = 1
    BoxNb[36, 37] = 1; BoxNb[36, 44:47] = 1
    BoxNb[37, 38] = 1; BoxNb[37, 45:47] = 1
    BoxNb[38, 39] = 1; BoxNb[38, 46] = 1
    BoxNb[39, 40] = 1
    BoxNb[40, 41] = 1
    BoxNb[41, 42] = 1
    BoxNb[43, 44] = 1
    BoxNb[44, 45] = 1; BoxNb[44, 47] = 1
    BoxNb[45, 46:48] = 1

    for i in range(Nbox - 1):
        for j in range(i + 1, Nbox):
            if BoxNb[i, j] == 1:
                BoxNb[j, i] = 1

    return BoxNb