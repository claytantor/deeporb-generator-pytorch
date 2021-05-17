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


docker run --gpus all --shm-size=1g --ulimit \
   memlock=-1 --ulimit stack=67108864 -it --rm \
   -v $(pwd)/workspace:/workspace \
   claytantor/deeporb-generator-pytorch:latest python train.py \
   --data_dir /workspace/txt \
   --session beethoven_words_a \
   --training_dir /workspace/training \
   --number 4000

docker run --gpus all --shm-size=1g --ulimit    memlock=-1 --ulimit stack=67108864 -it --rm    -v $(pwd)/workspace:/workspace    claytantor/deeporb-generator-pytorch:latest python train.py    --data_dir /workspace/txt    --session beethoven_words    --training_dir /workspace/training    --number 4000


   docker run --gpus all --shm-size=1g --ulimit \
  memlock=-1 --ulimit stack=67108864 -it \
  --rm -v $(pwd)/workspace:/workspace \
  claytantor/deeporb-generator-pytorch:latest python encode_midi_words.py \
  -m /workspace/midi/midi_dubroom_org \
  -o /workspace/txt -s midi_dubroom_org


docker run --gpus all --shm-size=1g --ulimit   memlock=-1 --ulimit stack=67108864 -it   --rm -v $(pwd)/workspace:/workspace   claytantor/deeporb-generator-pytorch:latest python encode_midi_words.py   -m /workspace/midi/midi_dubroom_org   -o /workspace/txt -s midi_dubroom_org

docker run --gpus all --shm-size=1g --ulimit    memlock=-1 --ulimit stack=67108864 -it --rm    -v $(pwd)/workspace:/workspace    claytantor/deeporb-generator-pytorch:latest python train.py    --data_dir /workspace/txt    --session midi_dubroom_org    --training_dir /workspace/training    --number 10000


/workspace/midi/midi_dubroom_org/sure_dread_melodic_dub/11_Dis_Dread_Is_Dubbing.mid


docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace claytantor/deeporb-generator-pytorch:latest python predict.py --data_dir /workspace/txt -s midi_dubroom_org -t /workspace/training --midi_file /workspace/midi/midi_dubroom_org/sure_dread_melodic_dub/11_Dis_Dread_Is_Dubbing.mid -o /workspace/midi



Transformer
https://github.com/jason9693/MusicTransformer-pytorch

Lstm
https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21#:~:text=An%20LSTM%20has%20a%20similar,operations%20within%20the%20LSTM's%20cells.&text=These%20operations%20are%20used%20to,to%20keep%20or%20forget%20information.

https://www.youtube.com/watch?v=8HyCNIVRbSU&feature=emb_title

https://towardsdatascience.com/understanding-rnns-lstm-and-seq2seq-model-using-a-practical-implementation-of-chatbot-in-2b9ab76d1eda


docker run --gpus all --shm-size=1g --ulimit   memlock=-1 --ulimit stack=67108864 -it   --rm -v $(pwd)/workspace:/workspace   claytantor/deeporb-generator-pytorch:latest python encode_midi_words.py   -m /workspace/midi/collectionB_ClassicalArchives  -o /workspace/txt -s collectionB_ClassicalArchives

docker run --gpus all --shm-size=1g --ulimit    memlock=-1 --ulimit stack=67108864 -it --rm  -v $(pwd)/workspace:/workspace  claytantor/deeporb-generator-pytorch:latest python train.py  --data_dir /workspace/txt  --session collectionB_ClassicalArchives  --training_dir /workspace/training  --number 20000


--- 
docker run --gpus all --shm-size=1g --ulimit   memlock=-1 --ulimit stack=67108864 -it   --rm -v $(pwd)/workspace:/workspace   claytantor/deeporb-generator-pytorch:latest python encode_midi_words.py   -m /workspace/midi/classical_b/sample_a  -o /workspace/txt -s sample_a

docker run --gpus all --shm-size=1g --ulimit    memlock=-1 --ulimit stack=67108864 -it --rm --net=host -v $(pwd)/workspace:/workspace  claytantor/deeporb-generator-pytorch:latest python train.py  --data_dir /workspace/txt  --session sample_a  --training_dir /workspace/training  --number 4000 --push_host localhost:9091 --push_user admin --push_password Yazz23!

---- transformer

docker run --gpus all --shm-size=1g --ulimit    memlock=-1 --ulimit stack=67108864 -it --rm --net=host -v $(pwd)/workspace:/workspace  claytantor/deeporb-generator-pytorch:latest python train_transformer.py  

# trombone, pizacatto_strings, timpani, church_organ, chello, chior_aahs, synth_strings, trumpet, viola, orchestral_harp, violin

docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace claytantor/deeporb-generator-pytorch:latest python predict.py --data_dir /workspace/txt -s sample_b -t /workspace/training --midi_file "/workspace/midi/collectionB_ClassicalArchives/Greats/ca/Mozart/Viennese Sonatinas K439b n2 1mov.mid" -o /workspace/midi

