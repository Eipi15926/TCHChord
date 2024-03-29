import os
import pickle
import pandas as pd
from torch.utils.data import Dataset
import torch
from data_loader.constants import NOTES_TO_INT, CHORD_TO_INT
# from constants import NOTES_TO_INT, CHORD_TO_INT


class MidiDataset(Dataset):
    NUM_NOTES = max(NOTES_TO_INT.values()) + 1
    NOTES_TO_INT = NOTES_TO_INT
    CHORD_TO_INT = CHORD_TO_INT
    INTERSECT_THRESH = 2  # Number of notes to intersect to count as a chord

    SEQUENCE_LENGTH = 16
    USE_CHORD_ONEHOT = False  # 暂时按照binary先写

    def __init__(self, config):
        data_path = config['data_path']
        transform = config['transform']
        target_transform = config['target_transform']
        use_one_hot = config['use_one_hot']
        batch_len = config['batch_len']

        assert os.path.exists(data_path), "{} does not exist".format(data_path)
        if not data_path.endswith('.pkl'):
            raise IOError('{} is not a recoginizable file type (.pkl)'.format(data_path))

        # load data
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
            self.df = pd.DataFrame.from_dict(data_dict)
            self.transform = transform
            f.close()

        if use_one_hot:
            self.chord_converter = self.convert_binary_to_onehot

        self.path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.batch_len = batch_len

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 三层列表，曲子旋律小节，曲子略去
        melody = row['melody']
        chord = row['chords']
        
        if self.transform:
            melody = self.transform(melody)
        if self.target_transform:
            chord = self.target_transform(chord)
        if self.batch_len:
            melody = self.align(melody, self.batch_len)
            chord = self.align(chord, self.batch_len)
        
        melody = torch.FloatTensor(melody)
        chord = torch.FloatTensor(chord)

        return melody, chord

    def __len__(self):
        return len(self.df)

    @classmethod
    # todo
    def convert_binary_to_onehot(cls, chord):
        """Convert binary to onehot"""
        note_list = [0] * cls.NUM_NOTES
        notes = chord.split('.')

        for n in notes:
            note_list[cls.NOTES_TO_INT[n]] = 1

        return note_list

    @classmethod
    def align(cls, data, batch_len):
        if len(data) > batch_len:
            return data[0:batch_len]
        else:
            for i in range(batch_len - len(data)):
                data.append([1]*12)
            return data

if __name__ == '__main__':
    config = {'data_path': 'data/parse_output/train.pkl', 'batch_len':32, 'transform': None, 'target_transform': None, 'use_one_hot': False}
    train_dataset = MidiDataset(config)
    scoreboard = {}
    cnt = 0
    for data in train_dataset:
        if cnt % 1000 == 0:
            print(f'count: {cnt}')
        melody, chord = data
        length = len(melody)
        # index = int(length // 10)
        index = length
        if index in scoreboard.keys():
            scoreboard[index] += 1
        else:
            scoreboard[index] = 1
        cnt += 1
        
    print(scoreboard)