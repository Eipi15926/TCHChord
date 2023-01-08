import torch
import json
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config):
    
    # Data Parameter
    data_folder = config['data']['data_folder']
    parse_path = config['data']['parse_path']
    filename = config['data']['filename']
    USE_ONEHOT = config['data']['one-hot']

    # parse
    '''
    这段可以挪到gen_dataset里面去
    '''
    parse_file = os.path.join(parse_path, filename)
    if os.path.exists(parse_file):
        print(parse_file + ' already exists')
    else:
        print('parse data to ' + parse_file)
        parse(config['data'])
    
    dataset = gen_dataset(config['data']) # transfer output.pkl
    dataloader = gen_dataloader(dataset, config['dataloader'])
    model = gen_model(dataloader,
                      config=config['model'])
    model.train()

# predict chord output and evaluate model by predict.py

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default='config/config.json', type=str,
                           help='config file path (default: config/config.json)')
    args = parser.parse_args()

    if args.config:
        # load config file
        print(type(args.config))
        config = json.load(open(args.config))
    else:
        raise ValueError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    main(config)
