# cython: language_level=3

cimport cython
from libc.math cimport log2

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_check_sequence(list sequences, pattern, double threshold=0.95):
    cdef Py_ssize_t i, n = len(sequences)
    cdef int valid_count = 0
    cdef int total_count = 0
    cdef list invalid_indices = []

    for i in range(n):
        if sequences[i] is None:
            continue
        total_count += 1
        try:
            if pattern.fullmatch(sequences[i]):
                valid_count += 1
            else:
                invalid_indices.append(i)
        except:
            invalid_indices.append(i)

    if total_count == 0:
        return False, []

    cdef double ratio = <double>valid_count / <double>total_count

    if ratio == 1.0:
        return True, []
    elif ratio >= threshold:
        return True, invalid_indices

    return False, []


@cython.boundscheck(False)
@cython.wraparound(False)
def char_entropy(list values, set alphabet):
    cdef dict counts = {}
    cdef int total = 0
    cdef double entropy = 0.0
    cdef double freq
    cdef str seq, ch

    for seq in values:
        for ch in seq.upper():
            if ch in alphabet:
                counts[ch] = counts.get(ch, 0) + 1
                total += 1

    if total == 0:
        return 0.0

    for c in counts.values():
        freq = <double>c / <double>total
        entropy -= freq * log2(freq)

    return entropy

