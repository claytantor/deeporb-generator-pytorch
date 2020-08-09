import sys, os
import argparse
import string
import json
import music21 

from utils import parse_midi_notes, write_notes_model


def main(argv):
    print("starting midi to text encoding.")

    # Read in command-line parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--midi", action="store", required=True, dest="midi", help="midi file to open")

    parser.add_argument("-o", "--out", action="store", required=True, dest="out", help="output file to write to")

    args = parser.parse_args()

    notes_model = parse_midi_notes(args.midi)
    write_notes_model(notes_model, args.out)


if __name__ == "__main__":
    main(sys.argv[1:])