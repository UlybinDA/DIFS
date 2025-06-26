import numpy as np

from  services.hkl_pg_generation import generate_hkl_symmetry_cython

def _generate_hkl_by_pg(hkl_orig_array,general_positions):
    hkl_orig_array = hkl_orig_array.astype(np.float32)
    positions_array = np.stack(list(general_positions)).astype(np.float64)
    hkl, hkl_orig = generate_hkl_symmetry_cython(hkl_orig_array,positions_array)
    return hkl,hkl_orig
