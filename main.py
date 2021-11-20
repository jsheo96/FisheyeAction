# main.py
# execute our program with various options given by arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--camera', type='str', default='/video0')
# TODO: Add appropriate arguments for our program
args = parser.parse_args()

def main(args):
    # TODO: pass parsed arguments and execute our program step by step
    pass

if __name__ == '__main__':
    main(args)
