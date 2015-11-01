import os
import sys


def get_random_seed(num_bytes=4):
    seed = int.from_bytes(os.urandom(num_bytes), sys.byteorder)
    return seed
