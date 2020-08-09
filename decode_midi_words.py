import sys, os
import argparse
import string
import json
import music21 

from utils import parse_midi_notes, write_notes_model, make_dir, write_instrument_words, find_files

def char_range(c1, c2):
    """Generates the characters from `c1` to `c2`, inclusive."""
    for c in range(ord(c1), ord(c2)+1):
        yield chr(c)

def decode_words_to_notes(words, name, words_channel=1):
    words_all = words.split()
    notes_model = {}
    notes_model[name] = {}
    notes_model[name]['notes'] = []
    notes_model[name]['channel'] = words_channel
    for note_word in words_all:
        note_word_parts = note_word.split('_')
        # print(note_word_parts)
        note = {      
            'nameWithOctave': '{}{}'.format(note_word_parts[0], note_word_parts[1]),
            'duration':{
                'type': note_word_parts[2] if note_word_parts[2] != 'complex' else 'eighth' 
            }
        }
        notes_model[name]['notes'].append(note)

    return notes_model


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