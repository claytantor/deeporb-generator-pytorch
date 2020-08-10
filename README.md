# deeporb-generator-pytorch
Deep Orb is a project about computers learning how to make music. The core approach to that is Recurrant neural network based learning models are used to teach powerful GPU based systems to compose, choose instuments, genres and ultimately produce complete musical works.

# background


## LSTM (Long Short Term Memory) and RNNs (Recurrent Neural Networks)

* [Recurrent Neural Networks by Example in Python](https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470) 
* [TensorFlow Magenta](https://magenta.tensorflow.org/)

# setting up your GPU and container 
To use the [Nvidia Pytorch Docker Container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) your host computer will need to have the correct drivers installed first. 

## CUDA for pytorch
This is the CUDA and driver version I use on Ubuntu 20.04 for this project. It is strogly reccomended that you go [directly to NVidia downloads](https://developer.nvidia.com/cuda-downloads) and make sure you have the correct driver fully installed to enable using the container

when you are finsihed `nvidia-smi` should show you something like this:

```bash
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  On   | 00000000:2D:00.0  On |                  N/A |
|  0%   53C    P8    30W / 250W |    637MiB / 11016MiB |     16%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1372      G   /usr/lib/xorg/Xorg                 53MiB |
|    0   N/A  N/A      2220      G   /usr/lib/xorg/Xorg                214MiB |
|    0   N/A  N/A      2421      G   /usr/bin/gnome-shell              134MiB |
|    0   N/A  N/A      3005      G   ...AAAAAAAAA= --shared-files      208MiB |
|    0   N/A  N/A      5617      G   gnome-control-center                6MiB |
+-----------------------------------------------------------------------------+
```

[a script](./cuda_install.sh) is provided as an example and should work for Ubuntu 20.04 LTS

# building the container
This project exclusely uses docker containers that are provided with PyTorch and are Nvidia GPU and Driver enabled.

```
docker build -t pytorch-rnn-midi:latest .
```

# running the scripts

![pipeline](docs/img/DeepOrbPipeline2.svg)

## encode_midi_words.py - encoding midi to words
The use of the encoder is intended to allow for the translation of midi events into words that the LSTM Network can learn from.

```python
    parser.add_argument("-m", "--midi", action="store", required=True, dest="midi_dir", help="midi directory to open")

    parser.add_argument("-o", "--out", action="store", required=True, dest="out", help="output dir to write to")

    parser.add_argument("-s", "--session", action="store", required=True, dest="session", help="session name")
```

The three arguments are used to recursively scan through a directory for all miding files and segement them by midi instrument and then place note files in the heirchy by insrument under the session directory.

This will allow a dirctory with a collection of songs in a specific folder to be the source of training by all the shared instruments.

```bash
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python encode_midi_words.py -m /workspace/midi/early_pieces_6 -o /workspace/txt -s early_b
```

will generate JSON files that encode each track as a instrument in the tree hierchy under the session name:

```bash
early_b/
├── acoustic_grand_piano
│   ├── 96fa7371-cba0-4d61-8e64-884df4af5f93.txt
│   └── b01ba9e2-054c-4c3d-bfb1-4e16f19010f4.txt
├── brass_section
│   ├── brass_section-1aecb2.json
│   ├── brass_section-51cac2.json
│   └── brass_section-888501.json
├── fx_4_(atmosphere)
│   ├── fx_4_(atmosphere)-119656.json
│   └── fx_4_(atmosphere)-a5942e.json
├── fx_atmosphere
│   └── fx_atmosphere-7c194c.json
├── lead_6_(voice)
│   ├── lead_6_(voice)-153cd1.json
│   └── lead_6_(voice)-8d1aec.json
├── lead_8_(bass_+_lead)
│   ├── lead_8_(bass_+_lead)-1a3531.json
│   ├── lead_8_(bass_+_lead)-1e8e90.json
│   ├── lead_8_(bass_+_lead)-23c040.json
│   ├── lead_8_(bass_+_lead)-a2cb85.json
│   ├── lead_8_(bass_+_lead)-a7352d.json
│   └── lead_8_(bass_+_lead)-ab5ece.json
├── lead_bass_lead
│   ├── lead_bass_lead-65e195.json
│   ├── lead_bass_lead-dee892.json
│   └── lead_bass_lead-ed5419.json
├── lead_voice
│   └── lead_voice-3f3394.json
└── pad_2_(warm)
    ├── pad_2_(warm)-02008e.json
    └── pad_2_(warm)-05fbf5.json

```

eah file has a list of notes in common musical notation that will be turned into words for the LSTM to learn:

```json
{
    "notes": [{
        "nameWithOctave": "A3",
        "fullName": "A in octave 3 Quarter Note",
        "pitch": {
            "name": "A",
            "microtone": "(+0c)",
            "octave": "3",
            "step": "A"
        },
        "duration": {
            "type": "quarter"
        }
    }]
}
```



## train.py - use the instrument note files to train the model 


```bash
docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $(pwd)/workspace:/workspace pytorch-rnn-midi:latest python train.py --data_dir /workspace/txt/early_pieces_6_b --session musict1 --number 4000
```


