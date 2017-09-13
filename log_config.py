import logmatic
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger()


def setup_logger():
    # create file handler which logs even debug messages
    fh = RotatingFileHandler("data.log", mode='a', maxBytes=1024*1024,
                             backupCount=1000, encoding=None, delay=0)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    """ Options - 
    %(asctime) %(name) %(processName) %(filename) %(funcName) %(levelname) %(lineno) %(module) %(threadName) %(message)
    """
    formatter = logmatic.JsonFormatter(fmt="%(levelname) %(message)",
                                       extra={})  # process info in extra?
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)


def test_file_size_logging():
    for _ in range(3):
        logger.debug('DEBUG %(message)s', extra={'test_debug': 'test_debug1'})
        logger.info('INFO %(message)s', extra={'test_info': 'test_info2'})
        logger.warning('WARNING', extra={'test_warning': 'test_warning2'})
        logger.error('ERROR', extra={'test_error': 'test_error2'})
        logger.critical('CRITICAL', extra={'test_critical': 'test_critical2'})


def test_error_logging():
    # How to log errors
    try:
        a = 3 / 0
    except Exception:
        logger.error('Some kind of error', exc_info=True)


if __name__ == "__main__":
    setup_logger()
    test_file_size_logging()
    test_error_logging()
