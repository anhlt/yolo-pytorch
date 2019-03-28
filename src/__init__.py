from logging.config import dictConfig
import os

current_path = os.path.dirname(os.path.realpath(__file__))

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
            'filename': os.path.join(current_path, 'logs', 'server.log'),
            'mode': 'a',
            'formatter': 'detailed',
        },
        'errors': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(current_path, 'logs', 'server_errors.log'),
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
