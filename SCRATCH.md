# SCRATCH NOTES for deeporb-generator-pytorch
Deep Orb is a project about computers learning how to make music. The core approach to that is Recurrant neural network based learning models are used to teach powerful GPU based systems to compose, choose instuments, genres and ultimately produce complete musical works.


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

docker run --gpus all --shm-size=1g --ulimit \
  memlock=-1 --ulimit stack=67108864 -it \
  --rm nvcr.io/nvidia/pytorch:20.06-py3 /bin/sh


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace claytantor/deeporb-generator-pytorch:latest python predict.py --data_dir /workspace/txt/beethoven_words -s beethoven_words -t /workspace/training --midi_file /workspace/midi/beethoven/beeth3_2.mid

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace claytantor/deeporb-generator-pytorch:latest python predict.py --data_dir /workspace/txt -s beethoven_words -t /workspace/training --midi_file /workspace/midi/beethoven/rondo.mid -o /workspace/midi