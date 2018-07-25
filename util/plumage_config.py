import configparser
import json
from datetime import date
import os
def process_config(conf_file):
    """
    """
    params = {}
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(conf_file)
    for section in config.sections():
        if section == 'Directory':
            for option in config.options(section):
                params[option] = config.get(section, option)
        if section == 'DataSetHG':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Network':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Train':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Validation':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
        if section == 'Saver':
            for option in config.options(section):
                params[option] = eval(config.get(section, option))
    return params

def save_config(conf_file , save_dir):
    """
    Save the config file into params 
    """
    params = {}
    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(conf_file)
    for section in config.sections():
        if section == 'DataSetHG':
            params[section] = {}
            for option in config.options(section):
                params[section][option] = eval(config.get(section, option))
        if section == 'Network':
            params[section] = {}
            for option in config.options(section):
                params[section][option] = eval(config.get(section, option))
        if section == 'Train':
            params[section] = {}
            for option in config.options(section):
                params[section][option] = eval(config.get(section, option))
    if params['DataSetHG']['category'] is not None:
        file_name ="{}_{}_{}_config".format(str(date.today()), params['Network']['name'],
                                 params['DataSetHG']['category'])
    else:
        file_name ="{}_{}_all_config".format(str(date.today()), params['Network']['name'])
    file_path = os.path.join(save_dir, file_name)   
    print('Config saved in:',file_path)
    with open(file_path, 'w') as outfile:
        json.dump(params, outfile , indent=2)