import glob
import numpy as np
import sys, os
import argparse
import string
import json
import music21 
import pretty_midi
import random
import uuid
import re

from typing import Optional, List, Tuple, Dict, Union, Any
from collections import Counter
from math import floor
from pyknon.genmidi import Midi
from pyknon.music import NoteSeq, Note

from pretty_midi.constants import INSTRUMENT_MAP


def capitalize_all_words(all_words):
    capitalized = []
    words = all_words.split(' ')
    for word in words:
        capitalized.append(word.capitalize())
    return ' '.join(capitalized)

def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError:
        print("dir exists:", dir_name)

def find_files(dir_name, pattern="*.txt", recursive=True):
    # Using '*' pattern 
    all_files = [] 
    for name in glob.glob('{}/**/{}'.format(dir_name, pattern), recursive=True): 
        all_files.append(name)
    return all_files


def write_instrument_words(instument, words, base_dir):
    instrument_dir = '{}/{}'.format(base_dir,instument.replace(' ','_')).lower()
    make_dir(instrument_dir)
    file_name = '{}/{}.txt'.format(instrument_dir,str(uuid.uuid4()))
    with open(file_name, 'w+') as f:
        f.write(words)
    f.close()
    print("wrote", file_name)


def make_vocabulary_for_instrument(insrument_model, batch_size, seq_size):

    text = insrument_model['words']

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    # print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    
    if(len(in_text)>0 and len(out_text)>0):
        # print(in_text, out_text)

        out_text[:-1] = in_text[1:]
        out_text[-1] = in_text[0]
        in_text = np.reshape(in_text, (batch_size, -1))
        out_text = np.reshape(out_text, (batch_size, -1))
        return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text
    else:
        return int_to_vocab, vocab_to_int, n_vocab, [], []


def get_data_from_files(train_directory, session, batch_size, seq_size):

    #make the session dir under
    session_dir = '{}/{}'.format(train_directory,session)
    make_dir(session_dir)

    all_tracks = {}

    all_files = find_files(train_directory, pattern="*.json", recursive=True)


    # read all files to memory
    for train_file in all_files:
        ## assume the file is in an instrument dir
        instrument_key = train_file.split('/')[-2:-1][0]
        # print(instrument_key)
        # print(instrument_key, all_tracks)
        if(instrument_key not in all_tracks):
            # print("adding instrument")
            all_tracks[instrument_key] = {}
            all_tracks[instrument_key]['words'] = []

        print('reading file: {}'.format(train_file))
        with open(train_file, 'r') as f:
            t_f = f.read()

            ## this is a json file so make it a dictionary
            track_notes = json.loads(t_f)
            for note in track_notes['notes']:
                # print(note)
                all_tracks[instrument_key]['words'].append(note['word'])

    # print(all_tracks)

    delete_keys = []
    for instrument_key in all_tracks.keys():
        instrument_model = all_tracks[instrument_key]
        # print(instrument_key, instrument_model)
        int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = make_vocabulary_for_instrument(instrument_model, batch_size, seq_size)
        instrument_model['int_to_vocab'] = int_to_vocab
        instrument_model['vocab_to_int'] = vocab_to_int
        instrument_model['n_vocab'] = n_vocab

        if(len(in_text)== 0 or len(out_text) == 0):
            delete_keys.append(instrument_key)
        else:
            instrument_model['in_text'] = in_text
            instrument_model['out_text'] = out_text
    
    for d in delete_keys:
        all_tracks.pop(d, None)
    
    return all_tracks

def write_notes_model_json(notes_model, file_name):
    with open(file_name, 'w+') as f:
        f.write(json.dumps(notes_model))
    
    f.close()
    print("wrote", file_name)


def write_notes_model(notes_model, midi_file):
    # new_stream = stream.Stream()
    mf = music21.midi.MidiFile()
    track_index = 0
    for key in notes_model.keys():
        # print(notes_model[key])
        key_mt = music21.midi.MidiTrack(index=track_index)
        key_mt.setChannel(notes_model[key]['channel'])
        
        for note_item in notes_model[key]['notes']:
            midi_note = music21.note.Note(note_item['nameWithOctave'])
            midi_note.duration = music21.duration.Duration(note_item['duration']['type'])
            midi_note.volume.velocity = 120
            midiEvents = music21.midi.translate.noteToMidiEvents(midi_note)
            key_mt.events.extend(midiEvents)

        mf.tracks.append(key_mt)
        track_index += 1

    mf.open(midi_file, 'wb')
    mf.write()
    mf.close()
    print('wrote file:', midi_file)
  

def get_notes_list_from_track(midi_track):

    track_stream = music21.midi.translate.midiTrackToStream(midi_track)
    track_notes = get_notes_list_from_stream(track_stream)

    return track_notes


def get_notes_list_from_stream(midi_stream):  

    noteFilter=music21.stream.filters.ClassFilter('Note')
    # chordFilter=music21.stream.filters.ClassFilter('Chord')
    stream_notes = []

    for note in midi_stream.recurse().addFilter(noteFilter):
        note_dict = {
            'nameWithOctave': note.nameWithOctave,
            'fullName': note.fullName,
            'word': '{}_{}_{}'.format(note.pitch.name, str(note.pitch.octave), str(note.duration.type)).lower(),
            'pitch': {
                'name': note.pitch.name,
                'microtone': str(note.pitch.microtone),
                'octave': str(note.pitch.octave),
                'step': str(note.pitch.step)
            },
            'duration':{
                'type': str(note.duration.type)
            }
        }
        stream_notes.append(note_dict)
    
    return stream_notes

def parse_midi_notes(midi_fname):  

    mf=music21.midi.MidiFile()
    try:
        mf.open(midi_fname)
        mf.read()
        mf.close()
    except:
        print("Skipping file: Midi file has bad formatting")
        return
    
    tracks_all = []
    for track in mf.tracks:
        if(track.hasNotes()):

            # print(track.getProgramChanges())
            if(len(track.getProgramChanges())>0):
                track_model = {}
                i_name = pretty_midi.program_to_instrument_name(track.getProgramChanges()[0])
                track_model['notes'] = get_notes_list_from_track(track)
                track_model['name'] = i_name
                i_key = re.sub(r'[^A-Za-z ]', '', i_name)
                i_key = " ".join(i_key.split())
                track_model['key'] = i_key.replace(' ','_').lower()              
                track_model['program'] = track.getProgramChanges()[0]
                tracks_all.append(track_model)
    
    return tracks_all
                

def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])


def get_best_instrument_fast(instrument_name, instruments_map):

    if instrument_name == None:
        program_id_item = pretty_midi.instrument_name_to_program(instruments_map[0])
        item_class = pretty_midi.program_to_instrument_class(instruments_map[0])
        return {'name':instruments_map[0], 'program_id':program_id_item, 'class_name':item_class}
    elif instrument_name in instruments_map:
        program_id_item = pretty_midi.instrument_name_to_program(instrument_name)
        item_class = pretty_midi.program_to_instrument_class(program_id_item)
        return {'name':instrument_name, 'program_id':program_id_item, 'class_name':item_class}
    else:
        for iname in instruments_map:
            if instrument_name.lower() in iname.lower():
                program_id_item = pretty_midi.instrument_name_to_program(iname)
                item_class = pretty_midi.program_to_instrument_class(program_id_item)
                return {'name':iname, 'program_id':program_id_item, 'class_name':item_class}
            else:
                program_id_item = pretty_midi.instrument_name_to_program(instruments_map[0])
                item_class = pretty_midi.program_to_instrument_class(program_id_item)
                return {'name':iname, 'program_id':program_id_item, 'class_name':item_class}

def get_best_instrument(instrument_name, instruments_map):

    name_lookup = []
    # print(instrument_name)

    # program_id_instrument_name = pretty_midi.instrument_name_to_program(instrument_model['gm_name'])
    # instrument_class = pretty_midi.program_to_instrument_class(program_id_instrument_name)

    for iname in instruments_map:
        # print('instrument_name',instrument_name, iname)
        ratio = levenshtein_ratio_and_distance(instrument_name, iname, ratio_calc = True)

        program_id_item = pretty_midi.instrument_name_to_program(iname)
        item_class = pretty_midi.program_to_instrument_class(program_id_item)

        # only suggest if in the same class
        # if(instrument_class == item_class):
        #     name_lookup.append({'ratio':ratio,'name':iname})
        name_lookup.append({'ratio':ratio,'name':iname, 'program_id':program_id_item, 'class_name':item_class})

    # print(name_lookup)
    list_reverse = sorted(name_lookup, key=lambda instrument: instrument['ratio'], reverse=True)
    # value = random.randint(0, len(name_lookup)-1)
    return list_reverse[0]