import time
import pandas as pd
import numpy as np
from my_logger import mylogger
from encode_hkl import encode_hkl


CENTRINGS = ('P', 'A', 'B', 'C', 'I', 'F', 'Robv', 'Rrev', 'H')

PG_KEYS = {'1': '1',
           '-1': '2',
           '2': '3',
           'm': '4',
           '2/m': '5',
           '222': '6',
           'mm2': '7.z',
           'm2m': '7.y',
           '2mm': '7.x',
           'mmm': '8',
           '4': '9',
           '-4': '10',
           '4/m': '11',
           '422': '12',
           '4mm': '13',
           '-4m2': '14.cp',
           '-42m': '14.dp',
           '4/mmm': '15',
           '3': '16',
           '-3': '17',
           '312': '18.d',
           '321': '18.c',
           '3m1': '19.c',
           '31m': '19.d',
           '-31m': '20.d',
           '-3m1': '20.c',
           '6': '21',
           '-6': '22',
           '6/m': '23',
           '622': '24',
           '6mm': '25',
           '-6m2': '26.cp',
           '-62m': '26.dp',
           '6/mmm': '27',
           '23': '28',
           'm-3': '29',
           '432': '30',
           '-43m': '31',
           'm-3m': '32',
           }

PG_GENS = {'1': (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),),  # 1

           '2': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),),  # -1

           '3': (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),),  # 2

           '4': (np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),),  # m

           '5': (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 2/m
                 np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                 np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])),

           '6': (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # 222
                 np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                 np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])),

           '7.z': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # mm2
                   np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                   np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])),

           '7.y': (np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),  # m2m
                   np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                   np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])),

           '7.x': (np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),  # 2mm
                   np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                   np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])),

           '8': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # mmm
                 np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                 np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                 np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                 np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                 np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                 np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])),

           '9': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 4
                 np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                 np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])),

           '10': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # -4
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]])),

           '11': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 4/m
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]])),

           '12': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 422
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])),

           '13': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 4mm
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),

           '14.dp': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # -42m
                     np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                     np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
                     np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                     np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                     np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                     np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),

           '14.cp': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # -4m2
                     np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                     np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
                     np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                     np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                     np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                     np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])),

           '15': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 4/mmm
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),

           '16': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # 3
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]])),

           '17': (np.array([[1, 1, 0], [-1, 0, 0], [0, 0, -1]]),  # -3
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, -1]])),

           '18.d': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # 312
                    np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[-1, 0, 0], [1, 1, 0], [0, 0, -1]]),
                    np.array([[1, 1, 0], [0, -1, 0], [0, 0, -1]]),
                    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),),

           '18.c': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # 321
                    np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                    np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]]),
                    np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]]),),

           '19.c': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # 3m1
                    np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[-1, 0, 0], [1, 1, 0], [0, 0, 1]]),
                    np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]]),
                    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]])),

           '19.d': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # 31m
                    np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]]),
                    np.array([[1, 0, 0], [-1, -1, 0], [0, 0, 1]]),),

           '20.d': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # -31m
                    np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]]),
                    np.array([[1, 0, 0], [-1, -1, 0], [0, 0, 1]]),
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                    np.array([[0, -1, 0], [1, 1, 0], [0, 0, -1]]),
                    np.array([[1, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                    np.array([[1, 1, 0], [0, -1, 0], [0, 0, -1]]),
                    np.array([[-1, 0, 0], [1, 1, 0], [0, 0, -1]]),),

           '20.c': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # -3m1
                    np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                    np.array([[-1, 0, 0], [1, 1, 0], [0, 0, 1]]),
                    np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]]),
                    np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                    np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                    np.array([[0, -1, 0], [1, 1, 0], [0, 0, -1]]),
                    np.array([[1, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                    np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]]),
                    np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]]),
                    np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),),

           '21': (np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 6
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]])),

           '22': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # -6
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, -1]])),

           '23': (np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 6/m
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, -1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, -1]]),),

           '24': (np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 622
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [1, 1, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[1, 1, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]])),

           '25': (np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 6mm
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [1, 1, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]])),

           '26.cp': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # -6m2
                     np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                     np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                     np.array([[-1, -1, 0], [1, 0, 0], [0, 0, -1]]),
                     np.array([[0, 1, 0], [-1, -1, 0], [0, 0, -1]]),
                     np.array([[-1, 0, 0], [1, 1, 0], [0, 0, 1]]),
                     np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]]),
                     np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                     np.array([[-1, 0, 0], [1, 1, 0], [0, 0, -1]]),
                     np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                     np.array([[1, 1, 0], [0, -1, 0], [0, 0, -1]])),

           '26.dp': (np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),  # -62m
                     np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                     np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                     np.array([[-1, -1, 0], [1, 0, 0], [0, 0, -1]]),
                     np.array([[0, 1, 0], [-1, -1, 0], [0, 0, -1]]),
                     np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                     np.array([[1, 0, 0], [-1, -1, 0], [0, 0, 1]]),
                     np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]]),
                     np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                     np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]]),
                     np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]])),

           '27': (np.array([[1, 1, 0], [-1, 0, 0], [0, 0, 1]]),  # 6/mmm
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [1, 1, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[1, 1, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [-1, -1, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[-1, -1, 0], [0, 1, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, -1, 0], [0, 0, -1]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[-1, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [1, 1, 0], [0, 0, -1]]),
                  np.array([[-1, 0, 0], [1, 1, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[1, 1, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [-1, -1, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[-1, -1, 0], [0, 1, 0], [0, 0, -1]])),

           '28': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 23
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])),

           '29': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # m-3
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]])),

           '30': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # 432
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                  np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                  np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
                  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                  np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])),

           '31': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # -43m
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                  np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                  np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
                  np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]),
                  np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
                  np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]])),

           '32': (np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),  # m-3m
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                  np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                  np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                  np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
                  np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                  np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]),
                  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
                  np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                  np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]]),
                  np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]]),
                  np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]]),
                  np.array([[0, -1, 0], [0, 0, -1], [-1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]]),
                  np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]]),
                  np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]]),
                  np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]),
                  np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                  np.array([[0, -1, 0], [1, 0, 0], [0, 0, -1]]),
                  np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1]]),
                  np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                  np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]]),
                  np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                  np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                  np.array([[0, 0, -1], [0, -1, 0], [1, 0, 0]]),
                  np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0]]),
                  np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]),
                  np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])),

           }

@mylogger('DEBUG',log_args=True)
def generate_orig_hkl_array(h, k, l, pg, centring):
    hkl_array = ()
    if pg == '1':  # 1
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(-k, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '2':  # -1
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(-k, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 0] >= 0) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        bool_arr2 = ((hkl_array[:, 0] > 0) | (hkl_array[:, 2] != 0) | (hkl_array[:, 1] >= 0)).reshape(-1, 1)
        bool_arr = bool_arr1 & bool_arr2
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '3':  # 2
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(-k, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 0] >= 0) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '4':  # m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '5':  # 2/m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 0] >= 0) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '6':  # 222
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 0] >= 0) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        bool_arr2 = ((hkl_array[:, 0] >= 0) | (hkl_array[:, 1] != 0)).reshape(-1, 1)
        boolarr = bool_arr1 & bool_arr2
        hkl_array = hkl_array[boolarr[:, 0]]
        # hkl = hkl[check[:, 0] == True].reshape(-1, 3)

    elif pg == '7.z':  # mm2
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '7.y':  # m2m
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(-k, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '7.x':  # 2mm
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '8':  # mmm
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '9':  # 4
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 0] != 0) | (hkl_array[:, 1] == 0)).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '10':  # -4
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 0] > 0) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        bool_arr2 = ((hkl_array[:, 0] >= 0) | (hkl_array[:, 1] != 0)).reshape(-1, 1)
        bool_arr = bool_arr1 & bool_arr2
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '11':  # 4 / m
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 0] != 0) | (hkl_array[:, 1] == 0)).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '12':  # 422
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 0] != 0) | (hkl_array[:, 1] == 0)).reshape(-1, 1)
        bool_arr2 = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= hkl_array[:, 1])).reshape(-1, 1)
        bool_arr = bool_arr1 & bool_arr2

        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '13':  # 4mm
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr = ((hkl_array[:, 0] - hkl_array[:, 1]) >= 0).reshape(-1, 1)
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '14.dp':  # -42m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 1]) <= 0).reshape(-1, 1)
        bool_arr = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= 0)).reshape(-1, 1)
        boolarr = boolarr & bool_arr

        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '14.cp':  # -4m2
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= hkl_array[:, 1])).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '15':  # 4/mmm
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr = ((hkl_array[:, 0] - hkl_array[:, 1]) >= 0).reshape(-1, 1)
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '16':  # 3
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(-k, 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 1] >= 0) | (hkl_array[:, 0] != 0)).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '17':  # -3
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(-k, 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 1] >= 0) | (hkl_array[:, 0] != 0)).reshape(-1, 1)
        bool_arr2 = ((np.abs(hkl_array[:, 1]) < hkl_array[:, 0]) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        bool_arr = bool_arr1 & bool_arr2
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '18.d':  # 312
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(-k, 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 1] >= 0) | (hkl_array[:, 0] != 0)).reshape(-1, 1)
        bool_arr2 = ((np.abs(hkl_array[:, 1]) <= hkl_array[:, 0]) | (hkl_array[:, 2] != 0)).reshape(-1, 1)
        bool_arr = bool_arr1 & bool_arr2
        hkl_array = hkl_array[bool_arr[:, 0]]


    elif pg == '18.c':  # 321
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(-k, 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr1 = ((hkl_array[:, 1] >= 0) | (hkl_array[:, 0] != 0)).reshape(-1, 1)
        bool_arr2 = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] * 2 >= np.abs(hkl_array[:, 1]))).reshape(-1, 1)
        bool_arr3 = ((hkl_array[:, 2] != 0) | (np.abs(hkl_array[:, 1]) * 2 >= hkl_array[:, 0])).reshape(-1, 1)
        bool_arr = bool_arr1 & bool_arr2 & bool_arr3
        hkl_array = hkl_array[bool_arr[:, 0]]


    elif pg == '19.d':  # 31m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((hkl_array[:, 1] - hkl_array[:, 0]) >= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 0] * 2 + hkl_array[:, 1]) >= 0).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '19.c':  # 3m1
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '20.d':  # -31m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((hkl_array[:, 1] - hkl_array[:, 0]) >= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 0] * 2 + hkl_array[:, 1]) >= 0).reshape(-1, 1)
        bool_arr = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= 0)).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2 & bool_arr
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '20.c':  # -3m1
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= hkl_array[:, 1])).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '21':  # 6
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 0] != 0) | (hkl_array[:, 0] == hkl_array[:, 1])).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '22':  # -6 3/m
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(-k, 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 1] >= 0) | (hkl_array[:, 0] != 0)).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '23':  # 6/m
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        bool_arr = ((hkl_array[:, 0] != 0) | (hkl_array[:, 0] == hkl_array[:, 1])).reshape(-1, 1)
        hkl_array = hkl_array[bool_arr[:, 0]]

    elif pg == '24':  # 622
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        # boolarr = ((hkl_array[:, 0] - hkl_array[:, 1]) >= 0).reshape(-1, 1)
        bool_arr1 = ((hkl_array[:, 0] != 0) | (hkl_array[:, 1] == 0)).reshape(-1, 1)
        bool_arr2 = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= hkl_array[:, 1])).reshape(-1, 1)
        boolarr =  bool_arr1 & bool_arr2
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '25':  # 6mm
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(-l, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr = ((hkl_array[:, 0] - hkl_array[:, 1]) >= 0).reshape(-1, 1)
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '26.cp':  # -6m2
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)

    elif pg == '26.dp':  # -62m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((hkl_array[:, 1] - hkl_array[:, 0]) >= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 0] * 2 + hkl_array[:, 1]) >= 0).reshape(-1, 1)
        # bool_arr = ((hkl_array[:, 2] != 0) | (hkl_array[:, 0] >= 0)).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2  # & bool_arr
        hkl_array = hkl_array[boolarr[:, 0]]


    elif pg == '27':  # 6/mmm
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr = ((hkl_array[:, 0] - hkl_array[:, 1]) >= 0).reshape(-1, 1)
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '28':  # 23
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 1]) <= 0).reshape(-1, 1)
        boolarr2 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 2]) <= 0).reshape(-1, 1)
        bool_arr = ((np.abs(hkl_array[:, 0]) != hkl_array[:, 1]) | (hkl_array[:, 1] == hkl_array[:, 2])).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2 & bool_arr
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '29':  # m-3
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((hkl_array[:, 0] - hkl_array[:, 1]) <= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 0] - hkl_array[:, 2]) <= 0).reshape(-1, 1)
        bool_arr = ((hkl_array[:, 0] != hkl_array[:, 1]) | (hkl_array[:, 1] == hkl_array[:, 2])).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2 & bool_arr
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '30':  # 432
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 1]) <= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 1] - hkl_array[:, 2]) >= 0).reshape(-1, 1)
        boolarr3 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 2]) <= 0).reshape(-1, 1)
        bool_arr1 = ((hkl_array[:, 1] != hkl_array[:, 2]) | (hkl_array[:, 0] >= 0)).reshape(-1, 1)
        bool_arr2 = ((np.abs(hkl_array[:, 0]) != hkl_array[:, 2]) | (hkl_array[:, 0] >= 0)).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2 & boolarr3 & bool_arr1 & bool_arr2
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '31':  # -43m
        h_array = np.arange(-h, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 1]) <= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 1] - hkl_array[:, 2]) >= 0).reshape(-1, 1)
        boolarr3 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 2]) <= 0).reshape(-1, 1)
        bool_arr1 = ((hkl_array[:, 0] != hkl_array[:, 1]) | (hkl_array[:, 1] == hkl_array[:, 2])).reshape(-1, 1)
        boolarr = boolarr1 & boolarr2 & boolarr3 & bool_arr1
        hkl_array = hkl_array[boolarr[:, 0]]

    elif pg == '32':  # m-3m
        h_array = np.arange(0, h + 1, 1)
        k_array = np.arange(0, k + 1, 1)
        l_array = np.arange(0, l + 1, 1)
        hkl_array = np.array(np.meshgrid(h_array, k_array, l_array)).T.reshape(-1, 3)
        boolarr1 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 1]) <= 0).reshape(-1, 1)
        boolarr2 = ((hkl_array[:, 1] - hkl_array[:, 2]) >= 0).reshape(-1, 1)
        boolarr3 = ((np.abs(hkl_array[:, 0]) - hkl_array[:, 2]) <= 0).reshape(-1, 1)

        boolarr = boolarr1 & boolarr2 & boolarr3
        hkl_array = hkl_array[boolarr[:, 0]]

    if centring == 'P':
        pass

    elif centring == 'A':
        centring_mask = ((hkl_array[:, 1] + hkl_array[:, 2]) % 2 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]

    elif centring == 'B':
        centring_mask = ((hkl_array[:, 0] + hkl_array[:, 2]) % 2 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]

    elif centring == 'C':
        centring_mask = ((hkl_array[:, 0] + hkl_array[:, 1]) % 2 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]

    elif centring == 'I':
        centring_mask = ((hkl_array[:, 0] + hkl_array[:, 1] + hkl_array[:, 2]) % 2 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]

    elif centring == 'F':
        centring_mask1 = ((hkl_array[:, 0] + hkl_array[:, 1]) % 2 == 0).reshape(-1, 1)
        centring_mask2 = ((hkl_array[:, 0] + hkl_array[:, 2]) % 2 == 0).reshape(-1, 1)
        centring_mask3 = ((hkl_array[:, 1] + hkl_array[:, 2]) % 2 == 0).reshape(-1, 1)
        centring_mask = centring_mask1 & centring_mask2 & centring_mask3
        hkl_array = hkl_array[centring_mask[:, 0]]

    elif centring == 'Robv':
        centring_mask = ((-hkl_array[:, 0] + hkl_array[:, 1] + hkl_array[:, 2]) % 3 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]


    elif centring == 'Rrev':
        centring_mask = ((hkl_array[:, 0] - hkl_array[:, 1] + hkl_array[:, 2]) % 2 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]

    elif centring == 'Hex':
        centring_mask = ((hkl_array[:, 0] - hkl_array[:, 1]) % 3 == 0).reshape(-1, 1)
        hkl_array = hkl_array[centring_mask[:, 0]]

    return hkl_array

@mylogger('DEBUG')
def generate_hkl_by_pg(hkl_orig_array,pg_key):
    start = time.time()
    hkl_array = hkl_orig_array.copy().reshape(-1, 3, 1)
    general_positions = PG_GENS[pg_key]
    hkl_orig_array = hkl_orig_array.reshape(-1, 3, 1)
    for position in general_positions:
        new_hkl = np.matmul(position, hkl_orig_array)
        hkl_array = np.concatenate((hkl_array, new_hkl), axis=0)
        pass
    hkl_array = hkl_array.reshape(-1, 3)
    num_of_gen_pos = len(general_positions)
    hkl_encoded = encode_hkl(hkl_array)
    hkl_encoded, indices = np.unique(hkl_encoded, axis=0, return_index=True)
    indices = indices.reshape(-1,1)
    original_hkl = np.tile(hkl_orig_array.reshape(-1, 3), (num_of_gen_pos + 1, 1))[indices[:,0]]
    hkl_array = hkl_array[indices[:,0]]

    return hkl_array, original_hkl, hkl_orig_array

def multiply_hkl_by_pg(hkl_orig_hkl,pg):
    pg_key = PG_KEYS[pg]
    general_positions = PG_GENS[pg_key]
    orig_hkl = hkl_orig_hkl[:,3:]
    hkl = hkl_orig_hkl[:,0:3].reshape(-1,3,1)
    hkl_array = hkl.copy()
    for position in general_positions:
        new_hkl = np.matmul(position, hkl)
        hkl_array = np.concatenate((hkl_array,new_hkl),axis=0)
    hkl_array = hkl_array.reshape(-1, 3)
    num_of_gen_pos = len(general_positions)
    original_hkl = np.tile(orig_hkl, (num_of_gen_pos + 1, 1))
    return hkl_array, original_hkl


def get_key(d, value):
    for key, val in d.items():
        if val == value:
            return key
