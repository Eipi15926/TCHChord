import torch
import json
import argparse
import os
from parse.parse import parse
from data_loader.classify import data_split
# from data_loader.dataset import MidiDataset
from data_loader.dataloader import MidiDataLoader
from model.model import gen_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config):
    
    # Data Parameter
    data_folder = config['data']['data_folder']
    classify_folder = config['data']['classify_folder']
    parse_path = config['data']['parse_path']
    train_scale = config['data']['train_scale']
    val_scale = config['data']['train_scale']
    test_scale = config['data']['test_scale']
    # USE_ONEHOT = config['data']['use_one_hot']

    # parse
    '''
    这段可以挪到gen_dataset里面去
    '''
    parse_file = os.path.join(parse_path, 'train.pkl')
    if os.path.exists(parse_file):
        print(parse_file + ' already exists')
    else:
        print('parse data to ' + parse_path)
        data_split(data_folder, classify_folder, train_scale, val_scale, test_scale)
        
        config_parse_train = config['data']
        config_parse_train['data_folder'] = os.path.join(config['data']['classify_folder'], 'train')
        config_parse_train['filename'] = 'train.pkl'
        parse(config_parse_train)
        
        config_parse_val = config['data']
        config_parse_val['data_folder'] = os.path.join(config['data']['classify_folder'], 'val')
        config_parse_val['filename'] = 'val.pkl'
        parse(config_parse_val)
        
        config_parse_test = config['data']
        config_parse_test['data_folder'] = os.path.join(config['data']['classify_folder'], 'test')
        config_parse_test['filename'] = 'test.pkl'
        parse(config_parse_val)
        
    
    # dataset = gen_dataset(config['data']) # transfer output.pkl dataloader直接调用dataset
    train_loader = MidiDataLoader(config['train_data_loader']['args'])
    verify_loader = MidiDataLoader(config['val_data_loader']['args'])
    test_loader = MidiDataLoader(config['test_data_loader']['args'])
    model = gen_model(train_loader,
                      verify_loader,
                      test_loader,
                      config=config['model'])
    model.Train()
    model.test()

# predict chord output and evaluate model by predict.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config/config.json', type=str,
                           help='config file path (default: config/config.json)')
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
    else:
        raise ValueError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    main(config)
