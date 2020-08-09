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
from utils import get_data_from_files
from torchutils import print_info

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


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def train(device, net, criterion, optimizer,  in_text, out_text, n_vocab, vocab_to_int,  int_to_vocab, flags, target_dir, iteration_count=200):

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

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, iteration_count),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
            
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
        required=False, dest="data_dir", help="Source training text directory") 
      
    args = parser.parse_args()
        
    session_dir = os.path.join(os.getcwd(), "/workspace/training/{}".format(args.session))
    make_dir(session_dir)

    checkpoint_path = "{}/checkpoint_pt".format(session_dir)
    make_dir(checkpoint_path)


    if not path.exists(args.data_dir):
        raise ValueError('cannot find input dir: {}'.format(args.data_dir))

    flags = Namespace(  
            train_dir=args.data_dir,
            seq_size=32,
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=['I', 'am'],
            predict_top_k=5,
            checkpoint_path=checkpoint_path,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_info(device)
    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_files(
        flags.train_dir, flags.batch_size, flags.seq_size)

    net = RNNModule(n_vocab, flags.seq_size,
                    flags.embedding_size, flags.lstm_size)

                    
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, 0.01)


    train(device, net, criterion, optimizer,  in_text, out_text, n_vocab, vocab_to_int, int_to_vocab, flags, checkpoint_path, iteration_count=int(args.number))


if __name__ == "__main__":
    main(sys.argv[1:]) 
