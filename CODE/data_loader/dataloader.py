from dataset import MidiDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

class MidiDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, training=True, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(MidiDataLoader, self).__init__(**self.init_kwargs)


if __name__ == '__main__':
    dataloader = MidiDataLoader(MidiDataset,64,False,1,0)