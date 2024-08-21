from consts import WRITE_DIR

import os
import datetime
from pathlib import Path


def script_abs_path():
    # get the script absolute path
    return os.path.dirname(os.path.abspath(__file__))


def safe_write_dir_path(write_dir):
    # path to the results dir
    write_path = os.path.join(script_abs_path(), write_dir)
    # create a dir if it does not exist yet
    Path(write_path).mkdir(exist_ok=True)
    return write_path


def datetime_csv_filename(file_name):
    time_now = datetime.datetime.now()
    date_time_str = time_now.strftime("%Y-%m-%d-%H%M")
    csv_filename = "{}_{}.csv".format(file_name, date_time_str)
    return csv_filename


def results_in_subdir_path(subdir, csv_filename):
    # path to the results dir
    Path(WRITE_DIR).mkdir(exist_ok=True)
    save_dir = os.path.join(WRITE_DIR, subdir)
    save_path = safe_write_dir_path(save_dir)
    csv_path = os.path.join(save_path, csv_filename)
    return csv_path


def result_csv_save_path(csv_filename, add_date=True):
    # path to the results dir
    results_dir = safe_write_dir_path(WRITE_DIR)

    filename = csv_filename
    if add_date:
        # add datetime to the filename
        filename = datetime_csv_filename(csv_filename)
    # form a path for the csv file
    save_path = os.path.join(results_dir, filename)
    return save_path
