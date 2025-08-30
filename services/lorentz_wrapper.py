from services.calc_lorentz_coefficient import calc_lorentz_coefficient


def calc_lorentz(rotation_axis, diff_vectors, rec_vectors, beam_vec):
    rotation_axis = rotation_axis.reshape(-1)
    assert len(beam_vec) == 3, 'wrong rotation axis format'
    dv_shape = diff_vectors.shape
    assert all([len(dv_shape) == 2, dv_shape[1] == 3]), 'wrong diffraction vectors format'
    rv_shape = rec_vectors.shape
    assert all([len(rv_shape) == 2, rv_shape[1] == 3]), 'wrong diffraction vectors format'
    beam_vec = beam_vec.reshape(-1)
    assert len(beam_vec) == 3, 'wrong primary beam format'

    return calc_lorentz_coefficient(rotation_axis,
                                    diff_vectors,
                                    rec_vectors,
                                    beam_vec
                                    )
