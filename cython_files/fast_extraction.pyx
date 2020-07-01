import numpy as np

cdef list get_inorder_sliced(int[:] tmp_idxs, int[:] sliced, long[:] available_idxs):
    cdef dict idxs_dict
    cdef list tuple_sliced

    idxs_dict = {k:v for v,k in enumerate(tmp_idxs)}
    tuple_sliced = [( idxs_dict[sliced[i]] , i)   for i in available_idxs]
    tuple_sliced.sort(key=lambda tup: tup[0])
    tuple_sliced = [x[1] for x in tuple_sliced]
    return tuple_sliced


cpdef extract_values_2(list idxs, int k, float[:] data, int[:] indices, int[:] indptr):

    cdef Py_ssize_t x_idxs = len(idxs)
    cdef int x
    cdef int t
    cdef int indptr_start
    cdef int indptr_end
    cdef float[:] row_data

    cdef long[:] available_idxs
    cdef list ordered_available

    res = np.zeros((x_idxs, k), dtype=float)
    cdef double[:, :] res_view = res

    for x in range(x_idxs):
        indptr_start = indptr[x]
        indptr_end = indptr[x+1]


        available_idxs = np.where(np.isin(indices[indptr_start : indptr_end], idxs[x]))[0]

        ordered_available = get_inorder_sliced(idxs[x], indices[indptr_start : indptr_end], available_idxs)

        row_data = data[indptr_start:indptr_end]
        for t in range(len(ordered_available)):
            res_view[x, t] = row_data[ordered_available[t]]

    return res