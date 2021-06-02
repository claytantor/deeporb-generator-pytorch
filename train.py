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

from torch.autograd import Variable
from collections import Counter
from argparse import Namespace
from utils import get_data_from_files
from torchutils import print_info
from prometheus import make_registry, make_gauge, set_gauge


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
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

DROPOUT_P = 0
class MusicRNN(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size, model = 'lstm', num_layers=1):
        super(MusicRNN, self).__init__()
        self.model = model
        self.n_vocab = n_vocab
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        
        self.embeddings = nn.Embedding(n_vocab, lstm_size)
        if self.model == 'lstm':
            self.rnn = nn.LSTM(embedding_size, lstm_size, num_layers)
        elif self.model == 'gru':
            self.rnn = nn.GRU(embedding_size, lstm_size, num_layers)
        else:
            raise NotImplementedError
        self.out = nn.Linear(self.lstm_size, self.n_vocab)
        self.drop = nn.Dropout(p=DROPOUT_P)
        

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))


    def forward(self, x, prev_state):
        embeds = self.embeddings(x)
        output, state = self.rnn(embeds, prev_state)
        logits = self.out(output)

        return logits, state

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    return criterion, optimizer


def train(device, 
    net, criterion, optimizer,  in_text, out_text, n_vocab, 
    vocab_to_int,  int_to_vocab, flags, target_dir, 
    t_gauge, g_registry, push_host, session_id, instrument_key,
    iteration_count=200):

    iteration = 0

    for e in range(iteration_count):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)
        
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward(retain_graph=True)

            # Update the network's parameters
            optimizer.step()

            # loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            optimizer.step()

            if iteration % 1000 == 0:
                print('Epoch: {}/{}'.format(e, iteration_count),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
                if push_host != None:
                    set_gauge(t_gauge, loss_value, "loss_{}_{}".format(session_id, instrument_key), 
                        g_registry, p_host=push_host)
            
            if iteration % 1000 == 0:
                # ''.join(predict(device, net, flags.initial_words, n_vocab,
                #         vocab_to_int, int_to_vocab, top_k=5))
                torch.save(net.state_dict(),
                           '{}/model-{}.pth'.format(target_dir, iteration))

def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError:
        print("dir exists:", dir_name)

def capitalize_all_words(all_words):
    capitalized = []
    words = all_words.split(' ')
    for word in words:
        capitalized.append(word.capitalize())
    return ' '.join(capitalized)

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--session", action="store",
        required=True,  dest="session", help="the sessionid for the training")    

    parser.add_argument("-n", "--number", action="store", default=200,
        required=False,  dest="number", help="the number of iterations")                                        
        
    parser.add_argument("-d", "--data_dir", action="store",
        required=False, dest="data_dir", help="Source text directory") 

    parser.add_argument("-t", "--training_dir", action="store",
        required=False, dest="training_dir", help="Training directory") 

    parser.add_argument("-m", "--push_host", action="store", default=None,
        required=False, dest="push_host", help="Host to push metrics to") 

    parser.add_argument("-u", "--push_user", action="store",
        required=False, dest="push_user", help="Push metric user") 

    parser.add_argument("-p", "--push_password", action="store",
        required=False, dest="push_password", help="Push metric password") 

    parser.add_argument("-r", "--learning_rate", action="store", default="0.001",
        required=False, dest="learning_rate", help="Learning rate") 
      
    args = parser.parse_args()
        
    session_dir = os.path.join(os.getcwd(), "{}/{}".format(args.training_dir, args.session))
    make_dir(session_dir)

    if not path.exists(args.data_dir):
        raise ValueError('cannot find input dir: {}'.format(args.data_dir))

    flags = Namespace(  
            train_dir=args.data_dir,
            session_name=args.session,
            seq_size=16,
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=['I', 'am'],
            predict_top_k=5
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(device)

    data_dir = '{}/{}'.format(flags.train_dir, flags.session_name)
    all_tracks = get_data_from_files(
         data_dir, flags.batch_size, flags.seq_size)

    # print(all_tracks)

    if args.push_host != None:
        os.environ['PUSH_USER'] = args.push_user
        os.environ['PUSH_PASSWORD'] = args.push_password

    print('using metrics host', args.push_host)
    
    g_registry = make_registry()
    t_gauge = make_gauge('job_deeporb_training', "The training data for the deeporb ", g_registry)
    
    for instrument_key in all_tracks.keys():
        print("\n ===training model for instrument: {}".format(instrument_key))
        instrument_track = all_tracks[instrument_key]


        checkpoint_path = "{}/{}/checkpoint_pt".format(session_dir, instrument_key)
        make_dir(checkpoint_path)

        # our network
        # net = MusicRNN(instrument_track['n_vocab'], flags.seq_size,
        #                 flags.embedding_size, flags.lstm_size)  

           
        net = RNNModule(instrument_track['n_vocab'], flags.seq_size,
                        flags.embedding_size, flags.lstm_size)  
        net = net.to(device)

        # or optimizer
        criterion, optimizer = get_loss_and_train_op(net, float(args.learning_rate))

        # run tranining
        train(device, net, criterion, optimizer,  instrument_track['in_text'], instrument_track['out_text'],  
            instrument_track['n_vocab'], instrument_track['vocab_to_int'], instrument_track['int_to_vocab'], 
            flags, checkpoint_path, 
            t_gauge, g_registry, args.push_host, flags.session_name, instrument_key,
            iteration_count=int(args.number))


if __name__ == "__main__":
    main(sys.argv[1:]) 
