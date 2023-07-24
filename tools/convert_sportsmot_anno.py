# convert mot to trackingNet format
import os
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

DATA_PATH='datasets/SportsMOT'
SPLITS=['train','val']
OUT_PATH = os.path.join(DATA_PATH, 'tracking_annos')

for split in SPLITS:
    out_path = os.path.join(OUT_PATH,split)
    mkdir(out_path)
    for seq in tqdm(os.listdir(os.path.join(DATA_PATH,split))):
        # read gt of all players
        with open(os.path.join(DATA_PATH,split,seq,'gt','gt.txt'),'r') as f:
            gt=f.readlines()
        
        # key: id, value: gt of player(id)
        players={}
        for line in gt:
            line=line.split(',')
            frame, id = map(int, line[:2])
            x,y,w,h = map(int, line[2:6])
            if id not in players.keys():
                players[id]=[]
            players[id].append((frame,x,y,w,h))

        # output, there is no need to sort using `frame`
        length=len(os.listdir(os.path.join(DATA_PATH,split,seq,'img1')))
        
        for id in players.keys():
            with open(os.path.join(out_path,f'{seq}-{id:0>3d}.txt'),'w') as f:
                # start from frame 1
                i=0
                for cur in range(1,length+1):
                    try:
                        frame,x,y,w,h=players[id][i]
                    except:
                        i=cur
                        break
                    if frame != cur:
                        f.write('0,0,0,0\n')
                    else:
                        f.write(f'{x},{y},{w},{h}\n')
                        i+=1
                while i<=length:
                    f.write('0,0,0,0\n')
                    i+=1
