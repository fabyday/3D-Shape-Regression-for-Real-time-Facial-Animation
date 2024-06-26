import logging

import logging.handlers
import os 

thread_logger = logging.getLogger("workerLogger")
thread_logger.setLevel(logging.DEBUG)
# RotatingFileHandler
log_max_size = 10 * 1024 * 1024  ## 10MB
log_file_count = 20
log_path = "./logs"
if not os.path.exists(log_path):
    os.makedirs(log_path)


rotatingFileHandler = logging.handlers.RotatingFileHandler(
    filename= os.path.join(log_path, 'output.log'),
    maxBytes=log_max_size,
    backupCount=log_file_count
)
thread_logger.addHandler(rotatingFileHandler)