"""Logging setup"""

import logging


def set_default_logging_format(level=None, include_timestamp=True,
                               include_function=True):
    """Generate decently looking logging format string."""

    if level is None:
        level = logging.INFO

    color = "\033[1;%dm"
    reset = "\033[0m"
    black, red, green, yellow, blue, magenta, cyan, white = [
        color % (30 + i) for i in range(8)]
    logformat = '['
    if include_timestamp:
        logformat += f'{red}%(asctime)s.%(msecs)03d{reset}:'
    logformat += (f'{yellow}%(filename)s{reset}'
                  f':{blue}%(lineno)s{reset}')
    if include_function:
        logformat += f':{green}%(funcName)s{reset}'
    logformat += f'] %(message)s'

    logging.basicConfig(
        format=logformat,
        level=level,
        datefmt='%H:%M:%S')
