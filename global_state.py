import pandas as pd
import numpy as np
import os
from symmetry.pointsymmetry import PG_KEYS, CENTRINGS
from experiment.experiment import Experiment

LOG_FILE = "app.log"
ZIPPED_LOG_FILE = "app_log.zip"

point_groups = [*PG_KEYS.keys()]
centrings = list(CENTRINGS)

exp1 = Experiment()

class ContentVariables:
    def __init__(self):
        self.df_matr = pd.DataFrame(
            {'a*': [1., 0., 0.],
             'b*': [0., 1., 0.],
             'c*': [0., 0., 1.]
             }).to_dict('records')
        self.df_parameters = pd.DataFrame({
            'a': [10., ],
            'b': [10., ],
            'c': [10., ],
            'alpha': [90., ],
            'beta': [90., ],
            'gamma': [90., ],
            'omega': [0., ],
            'chi': [0., ],
            'phi': [0., ]}).to_dict('records')
        self.df_matr_transform = pd.DataFrame(
            {'TM1': [1., 0., 0.],
             'TM2': [0., 1., 0.],
             'TM3': [0., 0., 1.]
             }).to_dict('records')
        self.goniometer_table = pd.DataFrame(
            columns=('rotation', 'direction', 'name', 'real', 'angle')).to_dict('records')
        self.detector_geometry = pd.DataFrame(np.array([['', '', '', '']]),
                                              columns=('d_geometry', 'height', 'width', 'diameter')).to_dict('records')
        self.detector_geometry = pd.DataFrame(np.array([['']]), columns=('geometry',)).to_dict('records')
        self.detector_geometry_rectangle_parameters = pd.DataFrame(np.array([['', '']]),
                                                                   columns=('height_prm', 'width_prm')).to_dict(
            'records')
        self.detector_geometry_circle_parameters = pd.DataFrame(np.array([['']]),
                                                                columns=('diameter_prm',)).to_dict('records')
        self.complex_detector_parameters = pd.DataFrame(np.array([['', '', '', '']]),
                                                        columns=('rows_prm', 'columns_prm', 'row_spacing_prm',
                                                                 'column_spacing_prm')).to_dict('records')
        self.obstacles_parameters = pd.DataFrame(
            {'ditsance': [10, ], 'geometry': ['circle', ], 'orientation': ['normal', ],
             'rotations': [(0, 0, 0), ], 'lin_prm': [(0,)], 'displ y,z': [(0, 0)]}).to_dict('records')
        self.active_space_fig = None
        self.real_axes = None
        self.axes_angles = None


content_vars = ContentVariables()