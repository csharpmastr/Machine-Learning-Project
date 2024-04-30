import sys
from src.logger import logging

# create function for custom exception message
def error_msg_detail(error, error_detail:sys):
    
    # know which file an error has occur, what line, and etc
    _,_,exc_tb = error_detail.exc_info()
    
    # get filename where error has been occured in exc_tb
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # get line number where error has been occured in exc_tb
    line_num = exc_tb.tb_lineno
    
    # create custom error message
    err_msg = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        
        file_name, line_num, str(error)
    )
    
    return err_msg

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_msg_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
