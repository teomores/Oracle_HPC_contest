# tag: numpy_old
import numpy as np
#cimport numpy as np

def levenshtein(unsigned char[:] s1, unsigned char[:] s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    cdef int[:] current_row
    cdef int[:] previous_row = range(len(s2) + 1)
    cdef int insertions, deletions, substitutions

    for i, c1 in enumerate(s1):
        current_row = np.array([i + 1])
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
