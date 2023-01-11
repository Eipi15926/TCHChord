import os
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from constants import NOTES_TO_INT, CHORD_TO_INT, NOTES_TO_CHORD

class MidiDataset(Dataset):
    # NOTES = ['C', 'C#', 'D-', 'D', 'D#', 'E-', 'E', 'E#', 'F-', 'F', 'F#',
    #             'G-', 'G', 'G#', 'A-', 'A', 'A#', 'B-', 'B', 'B#', 'C-']
    # NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # NOTE_TO_INT = {k:v for v, k in enumerate(NOTES)}

    NUM_NOTES = max(NOTES_TO_INT.values()) + 1
    NOTES_TO_INT = NOTES_TO_INT
    CHORD_TO_INT = CHORD_TO_INT
    INTERSECT_THRESH = 2 # Number of notes to intersect to count as a chord

    SEQUENCE_LENGTH = 16
    USE_CHORD_ONEHOT = False # 暂时按照binary先写

    def __init__(self, path, transform=None,target_transform=None):
        assert os.path.exists(path), "{} does not exist".format(path)
        if not path.endswith('.pkl'):
            raise IOError('{} is not a recoginizable file type (.pkl)'.format(path))

        # load data
        with open(path, 'rb') as f:
            data_dict = pickle.load(f)
            self.df = pd.DataFrame.from_dict(data_dict)
            self.transform = transform
            f.close()
        
        if self.USE_CHORD_ONEHOT:
            self.chord_converter = self.convert_chord_to_onehot
        else:
            self.chord_converter = self.convert_chord_to_binary

        self.path=path
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        #三层列表，曲子旋律小节
        file =row['file']
        melody = [self.convert_note_to_int(n) for n in row['melody']]#todo
        chord = [self.chord_converter(c) for c in row['chords']]

        if self.transform:
            file = self.transform(file)
        if self.target_transform:
            melody = self.transform(melody)
            chord = self.target_transform(chord)
        return file, melody, chord

    def __len__(self):
        return len(self.df)

    @classmethod
    def convert_note_to_int(cls, note):
        return cls.NOTES_TO_INT[note]
        
    @classmethod
    def convert_chord_to_binary(cls, chord):
        """Convert chord to binary list"""
        note_list = [0] * cls.NUM_NOTES
        notes = chord.split('.')

        for n in notes:
            note_list[cls.NOTES_TO_INT[n]] = 1

        return note_list
    
    @classmethod
    def convert_chord_to_onehot(self, chord):
        """"Convert chord to onehot list"""
        chord_name = 'EMPTY'
        for k,v in NOTES_TO_CHORD.items():
            if len(set(chord.split('.')).intersection(set(k.split('.')))) >= self.INTERSECT_THRESH:
                chord_name = v
                break

        # offset by 1 to remove pad
        return self.CHORD_TO_INT[chord_name]

if __name__ == '__main__':
    data_path = 'output.pkl'
    dataset = MidiDataset(data_path)