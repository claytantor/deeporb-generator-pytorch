import sys, os
import os.path
from os import path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import glob
import random
import shutil

import numpy as np
from collections import Counter
import os
from argparse import Namespace
from utils import get_data_from_files, make_dir, parse_midi_notes, decode_words_to_notes, get_hash, write_notes_model_midi, load_yaml
from torchutils import print_info

class LSTMNetwork(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(LSTMNetwork, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]

def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return criterion, optimizer

def predict(device, net, initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    output = []
    # print("initial_words", initial_words)
    for w in initial_words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    initial_words.append(int_to_vocab[choice])

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        initial_words.append(int_to_vocab[choice])

    return initial_words



#==================
def generate_midi(data_dir, session, midi_file, words_top_k, out_dir, training_dir, config):

    # print("midi_file=",midi_file, "out_dir=",out_dir)
    # get all the instruments under
    list_training_subfolders_with_paths = [f.path for f in os.scandir(training_dir) if f.is_dir()]
    #print("list_training_subfolders_with_paths=",list_training_subfolders_with_paths)

    list_data_subfolders_with_paths = [f.path for f in os.scandir(data_dir) if f.is_dir()]

    notes_model_unmapped = parse_midi_notes(midi_file)
    # print("notes_model_unmapped=", notes_model_unmapped)
    notes_model_alltracks = list(filter(lambda x: x['key'] in config['predict']['instruments'], notes_model_unmapped))
    # print("notes_model_alltracks=", notes_model_alltracks)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(device)
    
    notes_model = {}

    # predict for each track based on the sample
    for track_n in notes_model_alltracks:
        filtered_data_path = list(filter(lambda x: track_n['key'] in x, list_data_subfolders_with_paths))
        filtered_training_path = list(filter(lambda x: track_n['key'] in x, list_training_subfolders_with_paths))

        #print(filtered_data_path, filtered_training_path, track_n['key'])

        if(len(filtered_data_path)>0 and len(filtered_training_path)>0):
            track_words = list(map(lambda x : x['word'], track_n['notes']))

            checkpoint_path = "{}/checkpoint_pt".format(filtered_training_path[0])
            source_dir = "{}/source".format(filtered_training_path[0])

            flags = Namespace(  
                    train_dir=data_dir,
                    seq_size=16,
                    batch_size=16,
                    embedding_size=64,
                    lstm_size=64,
                    gradients_norm=5,
                    initial_words=track_words[0:16],
                    predict_top_k=int(words_top_k), 
                    checkpoint_path=checkpoint_path,
                )


            # print("initial_words",flags.initial_words)

            track_data = get_data_from_files(filtered_data_path[0], flags.batch_size, flags.seq_size)[track_n['key']]

            net = LSTMNetwork(track_data['n_vocab'], flags.seq_size,
                        flags.embedding_size, flags.lstm_size)

            list_of_files = glob.glob('{}/*'.format(checkpoint_path)) 
            latest_file = max(list_of_files, key=os.path.getctime)

            net.load_state_dict(torch.load(latest_file))
            net.eval()
            net = net.to(device)
            words = predict(device, net, flags.initial_words, track_data['n_vocab'],
                track_data['vocab_to_int'], track_data['int_to_vocab'], top_k=5)

            instr_notes_model = decode_words_to_notes(words, track_n['key'], words_channel=track_n['channel'], words_program=track_n['program'])

            notes_model['{}-{}'.format(track_n['key'], get_hash(8))] = instr_notes_model[track_n['key']]

        else:
            print("pass track instrument not trained", track_n['key'])

    # all tracks 
    if(out_dir):
        midi_file = '{}/{}-{}.mid'.format(out_dir, session, get_hash(8))
        #print(midi_file, notes_model)
        write_notes_model_midi(notes_model, midi_file)
        return midi_file

    else:
        print(notes_model)
        return None

#=============


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", action="store",
        required=False, dest="data_dir", help="Source training text directory") 

    parser.add_argument("-s", "--session", action="store",
        required=True,  dest="session", help="the sessionid for the training")    
                                           
    parser.add_argument("-f", "--midi_file", action="store",
        required=False, dest="midi_file", help="Source midi file used for multi instument influence") 

    parser.add_argument("-w", "--topk", action="store", default="5",
        required=False, dest="topk", help="Number of top K words")   
    
    parser.add_argument("-o", "--out_dir", action="store",
        required=False, dest="out_dir", help="save predictions to directory")   

    parser.add_argument("-t", "--training_dir", action="store",
        required=True, dest="training_dir", help="Training directory") 
      
    args = parser.parse_args()

    config = load_yaml('./config.yml')['deeporb']
   
    training_dir = "{}/{}".format(args.training_dir, args.session)
    data_dir = "{}/{}".format(args.data_dir, args.session)

    generate_midi(data_dir, args.session, args.midi_file, args.topk, args.out_dir, training_dir, config)


if __name__ == "__main__":
    main(sys.argv[1:]) 
