import logging
from logging.config import fileConfig
import coloredlogs
from .version import __version__

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"

FIELD_STYLE = {
        'asctime'    : {'bright': True, 'color': 'black'}, 
        'levelname'  : {'color': 'black'}, 
        'name'       : {'color': 'blue'}
    }
    
LEVEL_STYLE = {
    'debug'   : {'color': 'white'},
    'info'    : {'color': 'green'},  
    'warning' : {'color': 'yellow'},
    'error'   : {'color': 'red'}, 
    'critical': {'bold': True, 'color': 'red'}
}

def configureLogger(name, filepath):
    fileConfig(filepath)

    logger = logging.getLogger(name)

    coloredlogs.install(logger=logger, fmt=LOG_FORMAT, \
        milliseconds=True, level_styles=LEVEL_STYLE, field_styles=FIELD_STYLE, level='info') 
    
    logging.getLogger(name).info(f"SYDR program initialized, version {__version__}")