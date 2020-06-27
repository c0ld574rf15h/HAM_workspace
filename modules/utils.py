import os
from configparser import ConfigParser


def read_config(section='DEFAULT'):
    BASE_URI = '/workspace' 
    config = ConfigParser()
    config.read(os.path.join(BASE_URI, 'resources/config.ini'))

    return config[section]


def wrap_path(filepath):
    BASE_URI = '/workspace/'

    return os.path.join(BASE_URI, filepath)
