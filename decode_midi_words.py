import sys, os
import argparse
import string
import json
import music21 

from utils import parse_midi_notes, write_notes_model, make_dir, write_instrument_words, find_files, decode_words_to_notes

def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)


def main(argv):
    print("starting midi to text encoding.")

    # Read in command-line parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in", action="store", required=True, dest="infile", help="input words")
    
    parser.add_argument("-n", "--name", action="store", required=True, dest="name", help="instrument name")

    parser.add_argument("-m", "--midi_dir", action="store", required=True, dest="midi_dir", help="midi directory to open")

    args = parser.parse_args()

    ## make or use dir
    make_dir(args.midi_dir)

    with open(args.infile, 'r') as f:
        t_f = f.read()
        notes_model = decode_words_to_notes(t_f, args.name)
        write_notes_model(notes_model, '{}/{}.mid'.format(args.midi_dir, args.name))


if __name__ == "__main__":
    main(sys.argv[1:])