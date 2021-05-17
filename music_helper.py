import music21
import pretty_midi
import re
import random
import argparse


all_instruments = [
    music21.instrument.Accordion,
    music21.instrument.AcousticBass,
    music21.instrument.AcousticGuitar,
    music21.instrument.Agogo,
    music21.instrument.Alto,
    music21.instrument.AltoSaxophone,
    music21.instrument.Bagpipes,
    music21.instrument.Banjo,
    music21.instrument.Baritone,
    music21.instrument.BaritoneSaxophone,
    music21.instrument.Bass,
    music21.instrument.BassClarinet,
    music21.instrument.BassDrum,
    music21.instrument.BassTrombone,
    music21.instrument.Bassoon,
    music21.instrument.BongoDrums,
    music21.instrument.BrassInstrument,
    music21.instrument.Castanets,
    music21.instrument.Celesta,
    music21.instrument.ChurchBells,
    music21.instrument.Clarinet,
    music21.instrument.Clavichord,
    music21.instrument.CongaDrum,
    music21.instrument.Contrabass,
    music21.instrument.Cowbell,
    music21.instrument.CrashCymbals,
    music21.instrument.Cymbals,
    music21.instrument.Dulcimer,
    music21.instrument.ElectricBass,
    music21.instrument.ElectricGuitar,
    music21.instrument.ElectricOrgan,
    music21.instrument.EnglishHorn,
    music21.instrument.FingerCymbals,
    music21.instrument.Flute,
    music21.instrument.FretlessBass,
    music21.instrument.Glockenspiel,
    music21.instrument.Gong,
    music21.instrument.Guitar,
    music21.instrument.Handbells,
    music21.instrument.Harmonica,
    music21.instrument.Harp,
    music21.instrument.Harpsichord,
    music21.instrument.HiHatCymbal,
    music21.instrument.Horn,
    music21.instrument.Kalimba,
    music21.instrument.KeyboardInstrument,
    music21.instrument.Koto,
    music21.instrument.Lute,
    music21.instrument.Mandolin,
    music21.instrument.Maracas,
    music21.instrument.Marimba,
    music21.instrument.MezzoSoprano,
    music21.instrument.Oboe,
    music21.instrument.Ocarina,
    music21.instrument.Organ,
    music21.instrument.PanFlute,
    music21.instrument.Percussion,
    music21.instrument.Piano,
    music21.instrument.Piccolo,
    music21.instrument.PipeOrgan,
    music21.instrument.PitchedPercussion,
    music21.instrument.Ratchet,
    music21.instrument.Recorder,
    music21.instrument.ReedOrgan,
    music21.instrument.RideCymbals,
    music21.instrument.SandpaperBlocks,
    music21.instrument.Saxophone,
    music21.instrument.Shakuhachi,
    music21.instrument.Shamisen,
    music21.instrument.Shehnai,
    music21.instrument.Siren,
    music21.instrument.Sitar,
    music21.instrument.SizzleCymbal,
    music21.instrument.SleighBells,
    music21.instrument.SnareDrum,
    music21.instrument.Soprano,
    music21.instrument.SopranoSaxophone,
    music21.instrument.SplashCymbals,
    music21.instrument.SteelDrum,
    music21.instrument.StringInstrument,
    music21.instrument.SuspendedCymbal,
    music21.instrument.Taiko,
    music21.instrument.TamTam,
    music21.instrument.Tambourine,
    music21.instrument.TempleBlock,
    music21.instrument.Tenor,
    music21.instrument.TenorDrum,
    music21.instrument.TenorSaxophone,
    music21.instrument.Timbales,
    music21.instrument.Timpani,
    music21.instrument.TomTom,
    music21.instrument.Triangle,
    music21.instrument.Trombone,
    music21.instrument.Trumpet,
    music21.instrument.Tuba,
    music21.instrument.TubularBells,
    music21.instrument.Ukulele,
    music21.instrument.UnpitchedPercussion,
    music21.instrument.Vibraphone,
    music21.instrument.Vibraslap,
    music21.instrument.Viola,
    music21.instrument.Violin,
    music21.instrument.Violoncello,
    music21.instrument.Vocalist,
    music21.instrument.Whip,
    music21.instrument.Whistle,
    music21.instrument.WindMachine,
    music21.instrument.Woodblock,
    music21.instrument.WoodwindInstrument,
    music21.instrument.Xylophone,
]

def build_model_from_music21_insstrument(music21_instrument):
    lookup_model = {}
    ival = music21_instrument()
    if ival.midiProgram != None:
        
        lookup_model[ival.midiProgram] = {}
        lookup_model[ival.midiProgram]['music21_instrument'] = ival
        lookup_model[ival.midiProgram]['class'] = pretty_midi.program_to_instrument_class(ival.midiProgram)

        gm_name = pretty_midi.program_to_instrument_name(ival.midiProgram)
        lookup_model[ival.midiProgram]['gm_name'] = gm_name 
        i_key = re.sub(r'[^A-Za-z ]', '', gm_name)
        i_key = " ".join(i_key.split())
        lookup_model[ival.midiProgram]['key'] = i_key.replace(' ','_').lower() 
    
    return lookup_model


def get_instruments():
    lookup_model = {}

    for music21_instrument in all_instruments:
        instrument_model = build_model_from_music21_insstrument(music21_instrument)
        for program_key in instrument_model.keys():
            lookup_model[program_key] = instrument_model[program_key]

    return lookup_model
            

def get_best_instrument_by_program(program_id, available_instruments, autoassign=True):

    if program_id in available_instruments.keys():
        return available_instruments[program_id]
    elif autoassign:
        random.choice(available_instruments)
    else:
        raise ValueError('program id not found')

def midiEventsToTempo(eventList):
    '''
    Convert a single MIDI event into a music21 Tempo object.
    TODO: Need Tests
    '''
    # from music21 import midi as midiModule
    # from music21 import tempo

    print(eventList[:4])
    if not music21.common.isListLike(eventList):
        event = eventList
    else:  # get the second event; first is delta time
        event = eventList[1]
    # get microseconds per quarter
    print(event.data)
    mspq = getNumber(event.data, 3)[0]  # first data is number
    bpm = round(60000000 / mspq, 2)
    # post = midiModule.getNumbersAsList(event.data)
    # environLocal.printDebug(['midiEventsToTempo, got bpm', bpm])
    mm = music21.tempo.MetronomeMark(number=bpm)
    return mm

def getNumber(midiStr, length):
    '''
    Return the value of a string byte or bytes if length > 1
    from an 8-bit string or bytes object
    Then, return the remaining string or bytes object
    The `length` is the number of chars to read.
    This will sum a length greater than 1 if desired.
    Note that MIDI uses big-endian for everything.
    This is the inverse of Python's chr() function.
    >>> midi.getNumber('test', 0)
    (0, 'test')
    Given bytes, return bytes:
    >>> midi.getNumber(b'test', 0)
    (0, b'test')
    >>> midi.getNumber('test', 2)
    (29797, 'st')
    >>> midi.getNumber(b'test', 4)
    (1952805748, b'')
    '''
    summation = 0
    if not music21.common.isNum(midiStr):
        for i in range(length):
            print(i, midiStr)
            midiStrOrNum = midiStr[i]
            if common.isNum(midiStrOrNum):
                summation = (summation << 8) + midiStrOrNum
            else:
                summation = (summation << 8) + ord(midiStrOrNum)
        return summation, midiStr[length:]
    else:  # midiStr is a number...
        midNum = midiStr
        summation = midNum - ((midNum >> (8 * length)) << (8 * length))
        bigBytes = midNum - summation
        return summation, bigBytes

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--action", action="store",
        required=False, dest="action", help="action type: plot") 

    parser.add_argument("-m", "--midi", action="store",
        required=False, dest="midi", help="The midi file") 
      
    args = parser.parse_args()

    print("hello world")


if __name__ == "__main__":
    main(sys.argv[1:]) 