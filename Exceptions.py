class RunsDictError(Exception):
    def __init__(self,modal):
        self.error_modal_content = modal

class CollisionError(Exception):
    def __init__(self,collision_report):
        self.error_modal_content = collision_report
        print(self.error_modal_content)

class InstrumentError(Exception):
    def __init__(self,modal):
        self.error_modal_content = modal

class HKLFormatError(Exception):
    def __init__(self,modal):
        self.error_modal_content = modal
