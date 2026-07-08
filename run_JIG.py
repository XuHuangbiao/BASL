import os

ml = ('CUDA_VISIBLE_DEVICES=0 nohup python -u main.py'
      ' --dataset JIG'
      ' --model-name JIG-S0'
      ' --action-type Suturing'
      ' --jig-fold 0'
      ' --video-path ./JIGSAWS/data/frames'
      ' --label-path ./JIGSAWS/data/info'
      ' --lr 2e-4'
      ' --stageA_epochs 10 --stageB_epochs 50 --stageC_epochs 100'
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.5'
      ' --frame-length 160 --batch 8'
      ' > JIG-S0.txt 2>&1 &')
os.system(ml)

ml = ('CUDA_VISIBLE_DEVICES=0 nohup python -u main.py'
      ' --dataset JIG'
      ' --model-name JIG-NP2'
      ' --action-type Needle_Passing'
      ' --jig-fold 2'
      ' --video-path ./JIGSAWS/data/frames'
      ' --label-path ./JIGSAWS/data/info'
      ' --lr 2e-4'
      ' --stageA_epochs 10 --stageB_epochs 50 --stageC_epochs 100'
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.5'
      ' --frame-length 160 --batch 8'
      ' > JIG-NP2.txt 2>&1 &')
os.system(ml)

ml = ('CUDA_VISIBLE_DEVICES=1 nohup python -u main.py'
      ' --dataset JIG'
      ' --model-name JIG-KT1'
      ' --action-type Knot_Tying'
      ' --jig-fold 1'
      ' --video-path ./JIGSAWS/data/frames'
      ' --label-path ./JIGSAWS/data/info'
      ' --lr 2e-4'
      ' --stageA_epochs 10 --stageB_epochs 50 --stageC_epochs 100'
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.5'
      ' --frame-length 160 --batch 8'
      ' > JIG-KT1.txt 2>&1 &')
os.system(ml)
