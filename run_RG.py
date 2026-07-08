import os

ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model-name Ball --action-type Ball --lr 2e-4 ' \
     '--stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 > Ball.txt 2>&1 & '
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model-name Clubs --action-type Clubs --lr 2e-4 ' \
     '--stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 > Clubs.txt 2>&1 & '
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model-name Hoop --action-type Hoop --lr 2e-4 ' \
     '--stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 > Hoop.txt 2>&1 & '
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model-name Ribbon --action-type Ribbon --lr 2e-4 ' \
     '--stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 > Ribbon.txt 2>&1 & '
os.system(ml)
