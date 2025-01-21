import logging

# Configure logging
logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S")

# Create an instance of logging
logger = logging.getLogger()