import logging

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S")

# Create a logger
logger = logging.getLogger()