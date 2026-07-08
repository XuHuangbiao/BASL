import os

ml = ('CUDA_VISIBLE_DEVICES=0 nohup python -u main.py'
      ' --dataset FineDiving'
      ' --model-name Fine10'
      ' --video-path ./FINADiving_MTL_256s/FineDiving'
      ' --label-path ./FINADiving_MTL_256s/info/fine-grained_annotation_aqa.pkl'
      ' --train-split ./FINADiving_MTL_256s/info/train_split.pkl'
      ' --test-split ./FINADiving_MTL_256s/info/test_split.pkl'
      ' --lr 2e-4 '
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.1'
      ' --frame-length 96 --batch 16'
      ' --usingDD'
      ' > Fine10.txt 2>&1 &')
os.system(ml)

ml = ('CUDA_VISIBLE_DEVICES=1 nohup python -u main.py'
      ' --dataset FineDiving'
      ' --model-name Fine20'
      ' --video-path ./FINADiving_MTL_256s/FineDiving'
      ' --label-path ./FINADiving_MTL_256s/info/fine-grained_annotation_aqa.pkl'
      ' --train-split ./FINADiving_MTL_256s/info/train_split.pkl'
      ' --test-split ./FINADiving_MTL_256s/info/test_split.pkl'
      ' --lr 2e-4 '
      ' --dropout 0.3 --seed 0 --seed_data 0 --labeled_ratio 0.2'
      ' --frame-length 96 --batch 16'
      ' --usingDD'
      ' > Fine20.txt 2>&1 &')
os.system(ml)
