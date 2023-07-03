import time
import contextlib


@contextlib.contextmanager
def timer(msg: str = None, logger=None):
    if not logger:
        from loguru import logger as loguru_logger
        logger = loguru_logger

    start = time.time()
    yield
    logger.debug(f'{msg}, used {round(time.time() - start,3)} s')