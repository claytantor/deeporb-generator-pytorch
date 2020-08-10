import sys, os
import argparse
import string
import json
import music21 
import uuid

from utils import parse_midi_notes, write_notes_model, make_dir, write_instrument_words, find_files, write_notes_model_json

def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)

def make_word(pitch_name, pitch_octive, duration_type):
    return '{}_{}_{}'.format(pitch_name, pitch_octive, duration_type).lower()

def main(argv):
    print("starting midi to text encoding.")

    # Read in command-line parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--midi", action="store", required=True, dest="midi_dir", help="midi directory to open")

    parser.add_argument("-o", "--out", action="store", required=True, dest="out", help="output dir to write to")

    parser.add_argument("-s", "--session", action="store", required=True, dest="session", help="session name")

    args = parser.parse_args()

    ## make or use dir
    make_dir(args.out)

    ## make session dir
    session_dir = '{}/{}'.format(args.out, args.session)
    make_dir(session_dir)
 
    all_midi = find_files(args.midi_dir, pattern="*.mid", recursive=True)
    if(len(all_midi)==0):
        print("no midi files found")

    for midi_file in all_midi:
        print('reading file:', midi_file)
        notes_model_alltracks = parse_midi_notes(midi_file)
        for track_notes in notes_model_alltracks:
            if(len(track_notes['notes'])>0):
                track_dir = '{}/{}'.format(session_dir, track_notes['key'])
                make_dir(track_dir)
                track_file_name = "{}/{}-{}.json".format(track_dir, track_notes['key'], str(uuid.uuid4()).replace('-','')[2:8])
                write_notes_model_json(track_notes, track_file_name)



if __name__ == "__main__":
    main(sys.argv[1:])