from SM_Runner import make_sm_runner_from_args
from sys import argv, stderr

if __name__ == '__main__':
    make_sm_runner_from_args().play_record(fps=120)