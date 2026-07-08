import os

ml = ('CUDA_VISIBLE_DEVICES=0 nohup python -u main.py'
      ' --dataset MTL-AQA'
      ' --model-name MTL40'
      ' --video-path /data/xhb/CoRe/MTL-AQA/new'
      ' --lr 1e-4'
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.4'
      ' --frame-length 103 --batch 16'
      ' --usingDD'
      ' > MTL4.txt 2>&1 &')
os.system(ml)

ml = ('CUDA_VISIBLE_DEVICES=1 nohup python -u main.py'
      ' --dataset MTL-AQA'
      ' --model-name MTL10'
      ' --video-path /data/xhb/CoRe/MTL-AQA/new'
      ' --lr 1e-4'
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.1'
      ' --frame-length 103 --batch 16'
      ' --usingDD'
      ' > MTL1.txt 2>&1 &')
os.system(ml)