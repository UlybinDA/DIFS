import os
import psutil
import re

from pyexpat.errors import messages

XRDoll_PATH = 'D:\\XRDoll\\XSourceProfile\\versions_20200803\\'
XRDoll_NAME = 'RecViewer_v1.21.exe'

class Reconstruction():
    def __init__(self,):
        self.default_name = True
        self.list_data = None
        self.reconstruction_type =None
        self.where_frame = None
        self.frame_range = None
        self.pack_size = None
        self.sum_n_frame_while_calc = None
        self.beam_center_pix = None
        self.beam_center_correction_pix = None
        self.detector_angles_pitch_roll_yaw_deg = None
        self.detector_distance_correction_mm = None
        self.zero_shift_2th_omega_chi_deg = None
        self.inc_angle_corr = None
        self.dtheta_deg_range = None
        self.layer_amount = None
        self.sigment_number = None
        self.resolution = None
        self.save_path = None
        self.file_name = None
        self.saving_format = None
        pass

    def parse_str_to_data(self,text):
        self.list_data = text.split('\n')
        self.reconstruction_type = self.parse(self.list_data,'type:')
        self.where_frame = self.parse(self.list_data,'where_frame')[0]
        self.frame_range = self.parse(self.list_data,'frame_range')
        self.pack_size = self.parse(self.list_data,'pack_size')
        self.sum_n_frame_while_calc = self.parse(self.list_data,'sum_n_frame_while_calc')
        self.beam_center_pix = self.parse(self.list_data,'beam_center_pix')
        self.beam_center_correction_pix = self.parse(self.list_data,'beam_center_correction_pix')
        self.detector_angles_pitch_roll_yaw_deg = self.parse(self.list_data,'detector_angles_pitch_roll_yaw_deg')
        self.detector_distance_correction_mm = self.parse(self.list_data,'detector_distance_correction_mm')
        self.zero_shift_2th_omega_chi_deg = self.parse(self.list_data,'zero_shift_2th_omega_chi_deg')
        self.inc_angle_corr = self.parse(self.list_data,'zero_shift_2th_omega_chi_deg')
        self.dtheta_deg_range = self.parse(self.list_data,'dtheta_deg_range')
        self.layer_amount = self.parse(self.list_data,'layer_amount')
        self.sigment_number = self.parse(self.list_data,'sigment_number')
        self.resolution = self.parse(self.list_data,'resolution')
        self.save_path = self.parse(self.list_data,'where_save')
        self.file_name = self.parse(self.list_data,'file_name')
        self.saving_format = self.parse(self.list_data,'saving_format')

        self.frame_range = self.str_list_to_int_list(self.frame_range)
        self.pack_size = self.str_list_to_int_list(self.pack_size)[0]
        self.sum_n_frame_while_calc = self.str_list_to_int_list(self.sum_n_frame_while_calc)[0]
        self.beam_center_pix = self.str_list_to_int_list(self.beam_center_pix)
        self.beam_center_correction_pix = self.str_list_to_float_list(self.beam_center_correction_pix)
        self.detector_angles_pitch_roll_yaw_deg = self.str_list_to_float_list(self.detector_angles_pitch_roll_yaw_deg)
        self.detector_distance_correction_mm = self.str_list_to_float_list(self.detector_distance_correction_mm)[0]
        self.zero_shift_2th_omega_chi_deg = self.str_list_to_int_list(self.zero_shift_2th_omega_chi_deg)
        self.inc_angle_corr = self.str_list_to_int_list(self.inc_angle_corr)[0]
        self.dtheta_deg_range = self.str_list_to_float_list(self.dtheta_deg_range)
        self.layer_amount = self.str_list_to_int_list(self.layer_amount)[0]
        self.sigment_number = self.str_list_to_int_list(self.sigment_number)
        self.resolution = self.str_list_to_int_list(self.resolution)
        self.saving_format = self.str_list_to_int_list(self.saving_format)[0]



    def str_list_to_int_list(self,list):
        if list is not None:
            new_list = [int(i) for i in list]
            return new_list
        else:
            return list

    def str_list_to_float_list(self,list):
        if list is not None:
            new_list = [float(i) for i in list]
            return new_list
        else:
            return list

    def parse(self,list_data,text):
        for data in list_data:
            if text in data:
                data_ = re.split(r'\bw{0,}:',data,maxsplit=1)[1]
                data_ = re.split(r'//w{,}',data_,maxsplit=1)[0]
                data_ = data_.split('\t')
                data_ = [ i for i in data_ if i !='']
                return data_
        return None

    def check_path(self,):
        isfirst =  os.path.isfile(self.where_frame)
        if not isfirst:
            return False,f'There are no such file at {self.where_frame}'




    def check_attributes(self):
        attr_list = [i for i in dir(rec1) if '__' not in i]
        check_list = [getattr(self, i) is not None for i in attr_list]
        all_good = False not in check_list
        if all_good:
            print('All obligatory attributes are not None, is nice!')
        else:
            messages = [f'{i} is None, please select' for i, b in zip(attr_list,check_list) if b is False]
            for message in messages:
                print(message)
            return all_good
        bool_ = self.check_path()
        all_good &= bool_








        pass




class Reconstruction_input():
    def __init__(self):

        with open('file') as file_:
            self.file = file_.read()

    def parse_txt_for_reconstructions(self):
        if self.file is not None:
            list_recs = self.file.split('//\n//')


def launch_XRDoll():
    os.startfile(XRDoll_PATH + XRDoll_NAME)


def kill_XRDoll():
    for process in (process for process in psutil.process_iter() if process.name() == XRDoll_NAME):
        process.kill()

if __name__ == '__main__':
    with open('test.txt') as file:
        str_data = file.read()
    rec1 = Reconstruction()
    rec1.parse_str_to_data(str_data)
    rec1.check_attributes()
    # attr_list = [i for i in dir(rec1) if '__' not in i]

    # print(attr_list)
