from collections import namedtuple


MODAL_TUPLE = namedtuple('Modal_content', 'header, body')

delete_runs_empty_error = MODAL_TUPLE(header='Delete scans error',
                                      body='There are no runs to delete, add some first')

add_runs_empty_error = MODAL_TUPLE(header='Set scan error',
                                   body='Some parameters are either empty or have invalid values. Note that No of scan, '
                                        'Det orientation, Det rotations xyz, Det dist and sweep should not be empty,'
                                        'Det dist and sweep also should not equal to zero ')

add_runs_empty_gon_error = MODAL_TUPLE(header='Set scan error',
                                       body='Some initial goniometer parameters are empty')

add_runs_empty_displ_error = MODAL_TUPLE(header='Set scan error',
                                         body='Goniometer orientation set to independent'
                                              ' while displace y/z parameters are empty')

calc_exp_min_max_error = MODAL_TUPLE(header='Calc experiment error',
                                     body='The maximum and minimum values are reversed')

calc_exp_wavelength_error = MODAL_TUPLE(header='Calc experiment error',
                                        body='Wavelength is not selected, please do')

calc_exp_no_scans_error = MODAL_TUPLE(header='Calc experiment error',
                                      body='There are no scans to calculate')

calc_exp_centring_error = MODAL_TUPLE(header='Calc experiment error',
                                      body='Centring is not selected')

calc_exp_pg_error = MODAL_TUPLE(header='Calc experiment error',
                                body='Point group is not selected')

calc_exp_no_det_warn = MODAL_TUPLE(header='Calc experiment warning',
                                   body='Detector model is not selected. 4π steradian diffraction is taken into account')

calc_collision_error = MODAL_TUPLE(header='Collisions warning',
                                   body='Beware, a collision might be there:\n'
)

show_rec_no_data_error = MODAL_TUPLE(header='Show reciprocal space error',
                                     body='Calculate experiment first')

show_rec_cell_volume_error = MODAL_TUPLE(header='Show reciprocal space error',
                                         body='The cell volume is too big (>64000 Å). Consider shorten linear cell parameters')

diff_map3d_axes_error = MODAL_TUPLE(header='Diffraction map error',
                                    body='To build 3D maps you need at least 3 moveable axes')

diff_map2d_axes_error = MODAL_TUPLE(header='Diffraction map error',
                                    body='To build 2D maps you need at least 2 moveable axes')

diff_map1d_axes_error = MODAL_TUPLE(header='Diffraction map error',
                                    body='To build 1D maps you need at least 1 moveable axes')

load_det_2geoms_error = MODAL_TUPLE(header='Load detector error',
                                      body='Selected both circle and rectangle geometry, choose one')

load_det_no_geoms_error = MODAL_TUPLE(header='Load detector error',
                                      body='No detector geometry in input. Please enter det_geom:circle/rectangle')

load_det_diameter_error = MODAL_TUPLE(header='Load detector error',
                                      body='Diameter value is either undefined or incorrect')

load_det_wrong_inp_structure = MODAL_TUPLE(header='Load detector error',
                                      body='Input file structure is not correct. Every parameter should be given as parameter=value.\n'
                                           'Each parameter must be specified on a separate line ')

load_det_hg_wd_error = MODAL_TUPLE(header='Load detector error',
                                      body='Height or width value is either undefined or incorrect')

load_det_comp_form_error = MODAL_TUPLE(header='Load detector error',
                                      body='Rows or cols value is either undefined or incorrect')

load_det_spacing_error = MODAL_TUPLE(header='Load detector error',
                                      body='row_spacing or col_spacing value is either undefined or incorrect')

load_det_error = MODAL_TUPLE(header='Load detector error',
                                      body='Some ')

download_no_obst_error = MODAL_TUPLE(header='Download obstacles error',
                                      body='No obstacles to download')

load_obstacles_error = MODAL_TUPLE(header='Load obstacles error',
                                      body='Some values of obstacles are invalid')

read_json_error = MODAL_TUPLE(header='File format error',
                                      body='An error occured during reading a json file')

load_wavelength_error = MODAL_TUPLE(header='Wavelength load error',
                                      body='Wavelength should be positive integer or float')

load_goniometer_error = MODAL_TUPLE(header='Load goniometer error',
                                    body='Some of goniometer values are invalid' )

load_instrument_error = MODAL_TUPLE(header='Load instrument error',
                                    body='There appeared to be problems with:\n')

download_runs_error_gon = MODAL_TUPLE(header='Download runs error',
                                body='The goniometer is not set, please do')

download_runs_error_runs = MODAL_TUPLE(header='Download runs error',
                                body='There are no runs, please provide some')

load_runs_axes_number_error = MODAL_TUPLE(header='Load runs error',
                                body="The number of axes in the runs does not match the number of axes of the selected goniometers")

load_runs_axes_angle_error = MODAL_TUPLE(header='Load runs error',
                                body="Fake axis angle doesn't match value for the selected goniometer")

check_dict_error_fake_scan = MODAL_TUPLE(header='Load runs error',
                                body="Fake axis marked as scan axis")

instrument_error_no_goniometer = MODAL_TUPLE(header='Instrument error',
                                body="The goniometer is not set, please do")

check_dict_axes_n_mismatch = MODAL_TUPLE(header='Load runs error',
                                body="Fake axis marked as scan axis")

check_dict_axes_repeating = MODAL_TUPLE(header='Load runs error',
                                body="Some axes are repeating")

check_dict_axes_out_of_range = MODAL_TUPLE(header='Load runs error',
                                body="Axes indices is out of range")

hkl_format_error = MODAL_TUPLE(header='Upload hkl error',
                               body=".hkl file corrupted")

no_scan_data_to_save_error = MODAL_TUPLE(header='hkl data save error',
                                         body='There is no data to save')

wrong_hkl_array_shape = MODAL_TUPLE(header='hkl array shape error',
                                         body='hkl array shape must be (x,3)')

separate_unique_common_error = MODAL_TUPLE(header='Separate unique and common reflections',
                                         body='There must be at least two runs.')

aperture_value_error =  MODAL_TUPLE(header='Diamond anvil error',
                                         body='Aperture should be in [0, 90] range')

anvil_normal_not_defined = MODAL_TUPLE(header='Diamond anvil error',
                                         body='Anvil normal is not defined or incorrect')


