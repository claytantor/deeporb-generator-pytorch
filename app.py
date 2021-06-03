from flask import Flask, request, jsonify, send_file
from utils import load_yaml
from predict import generate_midi
from fluidgen import DeepOrbSynth



app = Flask(__name__)
app.config.from_envvar('APP_CONFIG')
synth = DeepOrbSynth(app.config['FLUID_SF2'], max_velocity=100, num_tracks=3)

# load the deeporb yaml
config = load_yaml(app.config['DEEPORB_CONFIG'])['deeporb']

@app.route('/')
def hello_world():
    return 'Hello, Docker!'

@app.route('/config', methods=['GET'])
def get_config():
    # print(app.config)
    response = {
        'vars':{
            'ENV':app.config['ENV'],
            'DATA_DIR': app.config['DATA_DIR'], 
            'OUTPUT_DIR': app.config['OUTPUT_DIR'], 
            'SESSION_ID': app.config['SESSION_ID'], 
            'SOURCE_MIDI': app.config['SOURCE_MIDI'],
            'TRAINING_DIR': app.config['TRAINING_DIR']
        },
        'appconfig': config
    }
    return jsonify(response), 200

@app.route('/generate/midi', methods=['GET'])
def get_gen_midi():

    """
    --data_dir ./workspace/txt \
    -s sample_a 
    -t ./workspace/training \
    --midi_file ./workspace/midi/beethoven/rondo.mid 
    -o ./workspace/midi
    
    """

    training_dir = "{}/{}".format(app.config['TRAINING_DIR'],app.config['SESSION_ID'])
    data_dir = "{}/{}".format(app.config['DATA_DIR'], app.config['SESSION_ID'])

    gen_filename = generate_midi(
        data_dir, 
        app.config['SESSION_ID'], 
        app.config['SOURCE_MIDI'], 
        int(app.config['TOP_K']), 
        app.config['OUTPUT_DIR_MID'], 
        training_dir, config)

    # return jsonify({'filename':gen_filename}), 200
    response_file = open(gen_filename, 'rb')
    return send_file(response_file, 
        mimetype='audio/midi', 
        as_attachment=True, 
        attachment_filename=gen_filename.split('/')[-1] )


@app.route('/generate/wav', methods=['GET'])
def get_gen_wav():

    """
    --data_dir ./workspace/txt \
    -s sample_a 
    -t ./workspace/training \
    --midi_file ./workspace/midi/beethoven/rondo.mid 
    -o ./workspace/midi
    
    """

    training_dir = "{}/{}".format(app.config['TRAINING_DIR'],app.config['SESSION_ID'])
    data_dir = "{}/{}".format(app.config['DATA_DIR'], app.config['SESSION_ID'])

    # generate_midi(data_dir, session, midi_file, words_top_k, out_dir, training_dir, config)
    gen_filename = generate_midi(
        data_dir, 
        app.config['SESSION_ID'], 
        app.config['SOURCE_MIDI'], 
        int(app.config['TOP_K']), 
        app.config['OUTPUT_DIR_MID'], 
        training_dir, config)

    
    # ./workspace/midi/sample_b-ad55b78d.mid
    gen_file_part = gen_filename.split('/')[-1]
    out_file_wav = "{}/{}".format(app.config['OUTPUT_DIR_WAV'], gen_file_part.replace(".mid",".wav"))
    synth.mid_to_wav(gen_filename, out_file_wav)

    # # return jsonify({'filename':gen_filename}), 200
    response_file = open(out_file_wav, 'rb')
    return send_file(response_file, mimetype='audio/wav', as_attachment=True, attachment_filename=out_file_wav.split('/')[-1]  )
    
