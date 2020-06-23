import logging

import coloredlogs

logger = logging.getLogger("lrgwd")
logger.setLevel(logging.INFO)
coloredlogs.install(level='DEBUG', logger=logger)
