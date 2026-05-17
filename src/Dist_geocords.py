from __future__ import annotations

import math


def Dist_geocords(lat1, lon1, lat2, lon2):
    R = 6373.0
    fi1 = lat1 * 3.1416 / 180.0
    fi2 = lat2 * 3.1416 / 180.0
    psi1 = lon1 * 3.1416 / 180.0
    psi2 = lon2 * 3.1416 / 180.0
    inside = (math.sin(0.5 * (fi2 - fi1)) ** 2) + math.cos(fi2) * math.cos(fi1) * (math.sin(0.5 * (psi2 - psi1)) ** 2)
    L = 2.0 * R * math.asin(math.sqrt(inside))
    return 1000.0 * L
