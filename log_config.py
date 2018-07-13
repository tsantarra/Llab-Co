import logmatic
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger()


def setup_logger():
    # create file handler which logs even debug messages
    file_handler = RotatingFileHandler("out/data.log", mode='a', maxBytes=1024*1024,
                             backupCount=1000, encoding=None, delay=0)
    file_handler.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)

    """ Options - 
    %(asctime) %(name) %(processName) %(filename) %(funcName) %(levelname) %(lineno) %(module) %(threadName) %(message)
    """
    formatter = logmatic.JsonFormatter(fmt="%(levelname) %(message)",
                                       extra={})  # process additional info in extra
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)


def test_file_size_logging():
    for _ in range(3):
        logger.debug('my debug message', extra={'test_debug': 'test_debug1'})
        logger.info('my info message', extra={'test_info': 'test_info2'})
        logger.warning('my warning message', extra={'test_warning': 'test_warning2'})
        logger.error('my error message', extra={'test_error': 'test_error2'})
        logger.critical('my critical message', extra={'test_critical': 'test_critical2'})


def test_error_logging():
    # How to log errors
    try:
        a = 3 / 0
    except Exception:
        logger.error('Some kind of error WITH TRACE INFO', exc_info=True)


if __name__ == "__main__":
    setup_logger()
    test_file_size_logging()
    test_error_logging()
