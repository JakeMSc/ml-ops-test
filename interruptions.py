import os

import numpy as np


def detect_ec2_interruption():
    rand_int = np.random.randint(0, 1000)
    if rand_int >= 990:
        print("Interrupting EC2")
        return True
    else:
        return False


class EC2Interruption(Exception):
    pass


def touch_empty_file(filename):
    with open(filename, "a"):
        os.utime(filename, None)
