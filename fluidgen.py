import time
import fluidsynth
from midi2audio import FluidSynth
from music21 import *
import mido
import random

class DeepOrbSynth:
    def __init__(self, sf2_file, sample_rate=44100, max_velocity=90, num_tracks=3):
        self.fs = FluidSynth(sf2_file, sample_rate=sample_rate)
        self.max_velocity = max_velocity
        self.num_tracks = num_tracks

    def mid_to_wav(self, midi_file, out_file):
        mid = mido.MidiFile(midi_file)
        mid.tracks = random.choices(mid.tracks, k=self.num_tracks) 

        for i, track in enumerate(mid.tracks):
            track_new = mido.MidiTrack()
            for msg in track:
                if msg.type == 'note_on' or msg.type == 'note_off' and msg.velocity and msg.velocity>self.max_velocity:
                    msg.velocity=self.max_velocity

        mid.save("dostmp.midi")
        self.fs.midi_to_audio("dostmp.midi",out_file)
        

if __name__ == "__main__":

    synth = DeepOrbSynth("./workspace/sf2/Arachno.sf2")
    synth.mid_to_wav("./workspace/midi/sample_b-ad55b78d.mid","./workspace/out/wav/sample_b-ad55b78d.wav")

    # max_velocity = 90
    # fs = FluidSynth("./workspace/sf2/Arachno.sf2", sample_rate=44100)

    # mid = mido.MidiFile("./workspace/midi/sample_b-ad55b78d.mid")
    # mid.tracks = random.choices(mid.tracks, k=3) 

    # for i, track in enumerate(mid.tracks):
    #     track_new = mido.MidiTrack()
    #     for msg in track:
    #         if msg.type == 'note_on' or msg.type == 'note_off' and msg.velocity and msg.velocity>max_velocity:
    #              msg.velocity=max_velocity

    # mid.save("./workspace/midi/sample_b-ad55b78d-clean.mid")
    # fs.midi_to_audio("./workspace/midi/sample_b-ad55b78d-clean.mid","./workspace/out/wav/sample_b-ad55b78d.wav")

