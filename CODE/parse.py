from music21 import converter, note, chord, stream, meter, key, interval, pitch
import os
from tqdm import tqdm
import pickle

sharps_to_semitone = {
    0: 0, # C
    1: 5, # G
    2: -2, # D
    3: 3, # A
    4: -4, # E
    5: 1, # B
    6: 6, # F#
    7: -1, # C#
    -1: -5, # F
    -2: 2, # Bb
    -3: -3, # Eb
    -4: 4, # Ab
    -5: -1, # Db
    -6: 6, # Gb
    -7: 1 # Cb
}

def parse_one_midi(midi_file):
    
    midi = converter.parse(midi_file)
    # midi.show('text')
    melody_list = []
    chord_list = []
    
    # normalization by moving to C major
    times = midi.recurse().getElementsByClass(meter.TimeSignature)[0]
    k = midi.analyze('key')
    
    sharps = k.sharps
    semitone_offset = sharps_to_semitone[sharps]
    midi = midi.transpose(semitone_offset)
    
    # print(k)
    # if k.mode == 'minor':
    #     i = interval.Interval(k.relative.tonic, pitch.Pitch('C'))
    # else:
    #     i = interval.Interval(k.tonic, pitch.Pitch('C'))
    # # transpose the music using stored interval
    # midi = midi.transpose(i)
    # 上面这一段也行 但是只能分析大小调 不确定是否有其他调式存在
    
    # get current chords and melody for all measures in midi
    parts = midi.recurse(classFilter='Part')
    melody_part = parts[0]
    chords_part = parts[1]
    for m in melody_part.recurse(classFilter='Measure'):
        cur_notes = [0] * 12
        for n in m.recurse(classFilter='Note'):
            p = int(n.pitch.pitchClassString, 16) # n.pitch.pitchClassString is a hexadecimal integer, index
            cur_notes[p] = 1
        melody_list.append(cur_notes)
    # 这里只考虑当前小节有没有某个音 没有考虑音的数量和时值 后续可以更改
    for m in chords_part.recurse(classFilter='Measure'):
        cur_chord = [0] * 12
        if(m.recurse(classFilter='Chord')):
            ps = m.recurse(classFilter='Chord')[0].pitches # 只考虑小节内第一个和弦
            for p in ps:
                index = int(p.pitchClassString, 16)
                cur_chord[index] = 1
        chord_list.append(cur_chord)
        
    return melody_list, chord_list

def parse(config):
    data_folder = config['data_folder']
    parse_path = config['parse_path']
    filename = config['filename']
    all_melody = []
    all_chords = []
    all_files = []
    if os.path.exists(parse_path):
        print(parse_path + ' already exist.')
    else:
        os.makedirs(parse_path)
        print('create folder ' + parse_path)
    midi_list = os.listdir(data_folder)
    print('---Start parsing data---')
    for midi_file in tqdm(midi_list):
        filepath = os.path.join(data_folder, midi_file)
        try:
            melody_list, chord_list = parse_one_midi(filepath)
            all_files.append(midi_file)
            all_melody.append(melody_list)
            all_chords.append(chord_list)
        except:
            print(f'Cannot parse {midi_file}')

    # put into dictionary and send to pickle file
    harmony = {}
    harmony['file'] = all_files
    harmony['melody'] = all_melody
    harmony['chords'] = all_chords
    # print(harmony)
    
    pickle_file = os.path.join(parse_path, filename)
    with open(pickle_file, 'wb') as filepath:
        pickle.dump(harmony, filepath)

    print('MIDI Processing Completed.')
    print(f'Totally parse {len(all_files)} files.')
        
if __name__ == '__main__':
    config = {'data_folder': 'data/all_key', 'parse_path': 'data/parse_output', 'filename': 'output.pkl'}
    # config = {'data_folder': 'data/test_data', 'parse_path': 'data/parse_output', 'filename': 'output.pkl'}
    parse(config)
