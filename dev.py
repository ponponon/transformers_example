from loguru import logger

try:
    from fairseq.dataclass.configs import *
except Exception as error:
    logger.exception(error)