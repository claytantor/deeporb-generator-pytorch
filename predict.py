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
import uuid

import numpy as np
from collections import Counter
import os
from argparse import Namespace
from utils import get_data_from_files, make_dir
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

def predict(device, net, initial_words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)

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


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_dir", action="store",
        required=False, dest="data_dir", help="Source training text directory") 

    parser.add_argument("-i", "--initial", action="store", default="endp31 wait9 p16 p31 wait1 endp31 wait1",
        required=False, dest="initial", help="Initial words to seed") 

    parser.add_argument("-s", "--session", action="store",
        required=True,  dest="session", help="the sessionid for the training")    

    # parser.add_argument("-n", "--number", action="store", default=200,
    #     required=False,  dest="number", help="the number of iterations")                                        
        
    parser.add_argument("-f", "--file", action="store",
        required=False, dest="file", help="Source file") 

    parser.add_argument("-w", "--words", action="store", default="5",
        required=False, dest="words", help="Number of words")   
    
    parser.add_argument("-o", "--out_dir", action="store",
        required=False, dest="out_dir", help="save predictions to directory")   
      
    args = parser.parse_args()
   
    session_dir = "/workspace/training/{}".format(args.session)
    checkpoint_path = "{}/checkpoint_pt".format(session_dir)
    source_dir = "{}/source".format(session_dir)
    
    flags = Namespace(  
            train_dir=args.data_dir,
            seq_size=32,
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=args.initial.split(' '),
            predict_top_k=int(args.words),
            checkpoint_path=checkpoint_path,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(device)
    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_files(
        flags.train_dir, flags.batch_size, flags.seq_size)

    net = RNNModule(n_vocab, flags.seq_size,
                        flags.embedding_size, flags.lstm_size)


    list_of_files = glob.glob('{}/*'.format(checkpoint_path)) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    net.load_state_dict(torch.load(latest_file))
    net.eval()
    net = net.to(device)
    words = predict(device, net, flags.initial_words, n_vocab,
        vocab_to_int, int_to_vocab, top_k=5)

    if(args.out_dir):
        instance_id = str(uuid.uuid4()).replace('-','')[:8]
        text_file = open('{}/{}-{}.txt'.format(args.out_dir, args.session, instance_id), "w")
        n = text_file.write(' '.join(words))
        text_file.close()
    else:   
        print(words)

if __name__ == "__main__":
    main(sys.argv[1:]) 
