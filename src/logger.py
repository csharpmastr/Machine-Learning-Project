import logging
import os
from datetime import datetime

# create log file
log_file = f"{datetime.now().strftime('%m_%d__%Y_%H_%M_%S')}.log"

# create path to store log file in the current working directory
log_path = os.path.join(os.getcwd(),"logs", log_file)

# make directory of logs
os.makedirs(log_path, exist_ok=True)

log_file_path = os.path.join(log_path, log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO   
)