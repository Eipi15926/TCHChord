from model.metric import mapping, evaluation_simple, chord_list
from data_loader.dataloader import MidiDataLoader
import json
import torch

config_path = 'config/config.json'
config = json.load(open(config_path))
verify_loader = MidiDataLoader(config['val_data_loader']['args'])
lim = config['model']['metric']['lim']
lim_num = config['model']['metric']['lim_num']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load('save/mymodel.pt')

avg = 0
cnt = 0
for melody, chord in verify_loader:
    # melody=torch.tensor(melody).to(torch.float32)
    # chord=torch.tensor(chord).to(torch.float32)
    melody = melody.to(device)
    chord = chord.to(device)

    pred=model(melody)
    pred=pred.detach().reshape(-1,pred.shape[-1])
    tot = len(pred)
    for i in range(tot):
        index = mapping(pred[i], lim_num, lim) - 2
        pred[i] = chord_list[index]
        
    avg=avg+evaluation_simple(pred,chord, model.eval_config) #调用验证函数
    
    if cnt % 10 == 0:
        print(f'count: {cnt}')
    
    cnt=cnt+1
    
    # if cnt % 100 == 0:
    #     print(f'count: {cnt}')
    
print('Verify set average accuracy:',avg/cnt)