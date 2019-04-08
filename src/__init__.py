from logging.config import dictConfig
import os
from .config import LOG_PATH


d = {
    'version': 1,
    'formatters': {
        'detailed': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_PATH, 'server.log'),
            'mode': 'a',
            'formatter': 'detailed',
        },
        'errors': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOG_PATH, 'server_errors.log'),
            'mode': 'w',
            'level': 'ERROR',
            'formatter': 'detailed',
        },
    },
    'loggers': {
        'base_logger': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'errors']
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['console', 'errors']
    },
}

dictConfig(d)




