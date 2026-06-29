# cython: language_level=3

cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport fabs

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _sign(double x) noexcept nogil:
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _h_kern(double *zp, double *zm, int i, int j,
                           int np_, int nm, double eps) noexcept nogil:
    cdef double a = zp[i]
    cdef double b = zm[j]
    cdef double denom = a - b
    if fabs(denom) <= 2 * eps:
        return _sign(<double>(np_ - 1 - i - j))
    return (a + b) / denom


def fast_medcouple(cnp.ndarray[double, ndim=1] x_in):
    cdef int n = x_in.shape[0]
    if n < 3:
        return 0.0

    cdef double eps1 = np.finfo(np.float64).eps
    cdef double eps2 = np.finfo(np.float64).tiny

    # Step 1: Sort descending
    cdef cnp.ndarray[double, ndim=1] x = np.sort(x_in)[::-1].copy()

    # Step 1.2: Median
    cdef double median
    if n % 2 == 0:
        median = (x[n // 2 - 1] + x[n // 2]) / 2.0
    else:
        median = x[n // 2]

    cdef double x_eps = eps1 * (eps1 + fabs(median))

    if fabs(x[0] - median) < x_eps:
        return -1.0
    if fabs(x[n - 1] - median) < x_eps:
        return 1.0

    # Scale for numerical stability
    cdef double scale = 2.0 * max(x[0] - median, median - x[n - 1])
    if scale == 0:
        return 0.0

    # Step 2: Build Z+ and Z- by value comparison
    cdef list zp_list = []
    cdef list zm_list = []
    cdef int i, j

    for i in range(n):
        if x[i] >= median - x_eps:
            zp_list.append((x[i] - median) / scale)
        if x[i] <= median + x_eps:
            zm_list.append((x[i] - median) / scale)

    cdef int np_ = len(zp_list)
    cdef int nm = len(zm_list)

    if np_ == 0 or nm == 0:
        return 0.0

    cdef cnp.ndarray[double, ndim=1] zp = np.array(zp_list, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] zm = np.array(zm_list, dtype=np.float64)

    cdef double *zp_ptr = &zp[0]
    cdef double *zm_ptr = &zm[0]

    # Steps 3-4: Johnson & Mizoguchi iteration
    cdef cnp.ndarray[long, ndim=1] left = np.zeros(np_, dtype=np.int64)
    cdef cnp.ndarray[long, ndim=1] right = np.full(np_, nm - 1, dtype=np.int64)

    cdef long l_total = 0
    cdef long r_total = <long>np_ * <long>nm
    cdef long mc_idx = r_total // 2

    cdef long mid, r_tent_total, l_tent_total
    cdef double wm, wm_eps, h
    cdef int n_mid

    cdef cnp.ndarray[double, ndim=1] row_med = np.empty(np_, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] wts = np.empty(np_, dtype=np.float64)
    cdef cnp.ndarray[long, ndim=1] r_tent = np.empty(np_, dtype=np.int64)
    cdef cnp.ndarray[long, ndim=1] l_tent = np.empty(np_, dtype=np.int64)

    while r_total - l_total > np_:

        # Weighted median of row medians
        n_mid = 0
        for i in range(np_):
            if left[i] <= right[i]:
                mid = (left[i] + right[i]) // 2
                row_med[n_mid] = _h_kern(zp_ptr, zm_ptr, i, mid, np_, nm, eps2)
                wts[n_mid] = right[i] - left[i] + 1
                n_mid += 1

        if n_mid == 0:
            break

        order = np.argsort(row_med[:n_mid])
        cum = np.cumsum(wts[:n_mid][order])
        half = cum[n_mid - 1] / 2.0
        wm = row_med[order[0]]
        for i in range(n_mid):
            if cum[i] >= half:
                wm = row_med[order[i]]
                break

        wm_eps = eps1 * (eps1 + fabs(wm))

        # P[i]: right tentative border
        j = 0
        for i in range(np_ - 1, -1, -1):
            while j < nm and _h_kern(zp_ptr, zm_ptr, i, j, np_, nm, eps2) - wm > wm_eps:
                j += 1
            r_tent[i] = j - 1

        # Q[i]: left tentative border
        j = nm - 1
        for i in range(np_):
            while j >= 0 and _h_kern(zp_ptr, zm_ptr, i, j, np_, nm, eps2) - wm < -wm_eps:
                j -= 1
            l_tent[i] = j + 1

        r_tent_total = 0
        l_tent_total = 0
        for i in range(np_):
            r_tent_total += r_tent[i] + 1
            l_tent_total += l_tent[i]

        if mc_idx <= r_tent_total - 1:
            for i in range(np_):
                right[i] = r_tent[i]
            r_total = r_tent_total
        elif mc_idx > l_tent_total - 1:
            for i in range(np_):
                left[i] = l_tent[i]
            l_total = l_tent_total
        else:
            return wm

    # Step 5: Collect the remaining and select
    remaining = []
    for i in range(np_):
        for j in range(left[i], right[i] + 1):
            remaining.append(_h_kern(zp_ptr, zm_ptr, i, j, np_, nm, eps2))

    if len(remaining) == 0:
        return 0.0

    cdef cnp.ndarray[double, ndim=1] rem = np.array(remaining, dtype=np.float64)
    rem.sort()
    cdef long sel = mc_idx - l_total
    if sel < 0:
        sel = 0
    if sel >= len(rem):
        sel = len(rem) - 1

    return rem[sel]