# pytorch-rnn-midi


# venv
 1178  sudo apt-get install python3-venv
 1179  python3 -m venv venv
 1180  source venv/bin/activate
 1181  pip install --no-cache-dir -r requirements.txt


python encode_midi_words.py -m /media/nvme0n1p2/home/clay/Music/midi/collectionDubroom/dubroom/reggae.dub -s dub1 -o ./workspace/words

python trian -d ./workspace/words/dub1/acoustic_grand_piano -s dub1_gp -n 4000

## docker

docker build -t pytorch-rnn-midi:latest .

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python poc.py 

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python poc.py -m /workspace/midi/any.mid -o /workspace/out/any.png

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python midi-to-encoding.py -m /workspace/midi/early_pieces_6.midi -o /workspace/txt/early_pieces_6_b

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python midi-to-encoding.py -m /workspace/midi/any.mid -o /workspace/out/any.png


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python make_test_train.py 


## train
python train.py --mode train --file ./training/metal_song_titles/source/The-Collected-Works-of-HP-Lovecraft_djvu_poems_clean.txt --session metal03 --number 4000

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python train.py --mode train --file /workspace/training/ --session metal03 --number 4000


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python train.py --data_dir /workspace/txt/early_pieces_6_b --session musict1 --number 4000


wrote file: /workspace/txt/early_pieces_6_b/4_38_chords.txt
wrote file: /workspace/txt/early_pieces_6_b/4_38_notes.txt
wrote file: /workspace/txt/early_pieces_6_b/4_62_chords.txt
wrote file: /workspace/txt/early_pieces_6_b/4_62_notes.txt
wrote file: /workspace/txt/early_pieces_6_b/12_38_chords.txt
wrote file: /workspace/txt/early_pieces_6_b/12_38_notes.txt
wrote file: /workspace/txt/early_pieces_6_b/12_62_chords.txt
wrote file: /workspace/txt/early_pieces_6_b/12_62_notes.txt


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python predict.py --data_dir /workspace/txt/early_pieces_6_b --session musict1


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python train.py --data_dir /workspace/txt/early_pieces_6_b/notes --session musict1_notes2 --number 4000

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python predict.py --data_dir /workspace/txt/early_pieces_6_b/notes --session musict1_notes2 --out_dir /workspace/out

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python train.py -d /workspace/words/dub1/acoustic_grand_piano -s dub1_gp -n 4000

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python predict.py -d /workspace/words/dub1/acoustic_bass -s dub1_acoustic_bass --initial  "b_1_quarter b_1_complex f#_2_quarter b_1_quarter b_1_complex f#_2_quarter b_1_quarter b_1_quarter b_1_quarter c#_2_eighth d_2_eighth e_2_eighth f#_2_quarter f#_2_quarter c#_2_eighth f#_2_eighth b_1_quarter b_1_complex f#_2_quarter b_1_quarter b_1_quarter b_1_quarter c#_2_eighth d_2_eighth e_2_eighth f#_2_quarter f#_2_quarter c#_2_eighth f#_2_eighth b_1_quarter b_1_quarter b_1_quarter c#_2_eighth d_2_eighth e_2_eighth f#_2_quarter f#_2_quarter c#_2_eighth f#_2_eighth" 