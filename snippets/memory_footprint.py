'''
    Prints total amount of memory
    in megabytes (MB).

'''
import os
import psutil


def memory_footprint():
    '''
        Prints total amount of memory
        in megabytes (MB).

    '''
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


if __name__ == '__main__':
    print(memory_footprint(), 'MB')
