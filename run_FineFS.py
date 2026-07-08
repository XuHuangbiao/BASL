import os

ml = 'CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --video-path ./FINE_FS/video_features ' \
     '--train-label-path ./FINE_FS/annotation --test-label-path ./FINE_FS/annotation ' \
     '--model-name TES-1 --action-type TES --score-type SP --dataset FineFS --lr 2e-4 --clip-num 185 --labeled_ratio ' \
     '0.5 --stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 --seed_data 45 > TES1.txt 2>&1 & '
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --video-path ./FINE_FS/video_features ' \
     '--train-label-path ./FINE_FS/annotation --test-label-path ./FINE_FS/annotation ' \
     '--model-name PCS-1 --action-type PCS --score-type SP --dataset FineFS --lr 2e-4 --clip-num 185 --labeled_ratio ' \
     '0.5 --stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 --seed_data 4 > PCS1.txt 2>&1 & '
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --video-path ./FINE_FS/video_features ' \
     '--train-label-path ./FINE_FS/annotation --test-label-path ./FINE_FS/annotation ' \
     '--model-name TES-2 --action-type TES --score-type FS --dataset FineFS --lr 2e-4 --clip-num 185 --labeled_ratio ' \
     '0.5 --stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 --seed_data 29 > TES2.txt 2>&1 & '
os.system(ml)
ml = 'CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --video-path ./FINE_FS/video_features ' \
     '--train-label-path ./FINE_FS/annotation --test-label-path ./FINE_FS/annotation ' \
     '--model-name PCS-2 --action-type PCS --score-type FS --dataset FineFS --lr 2e-4 --clip-num 185 --labeled_ratio ' \
     '0.5 --stageA_epochs 30 --stageB_epochs 70 --stageC_epochs 140 --dropout 0.3 --seed_data 32 > PCS2.txt 2>&1 & '
os.system(ml)
