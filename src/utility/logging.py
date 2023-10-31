import logging as log
import sys

root = log.getLogger()
root.setLevel(log.INFO)

formatter = log.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler = log.StreamHandler(sys.stdout)
stream_handler.setLevel(log.DEBUG)
stream_handler.setFormatter(formatter)

root.addHandler(stream_handler)
