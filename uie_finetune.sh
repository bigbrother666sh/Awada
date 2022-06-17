cd uie
#split the data
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type "ext" \
    --save_dir ./data \
    --splits 0.7 0.2 0.1
#finetune
python finetune.py \
    --train_path "./data/train.txt" \
    --dev_path "./data/dev.txt" \
    --save_dir "./checkpoint" \
    --learning_rate 1e-5 \
    --batch_size 6 \
    --max_seq_len 512 \
    --num_epochs 100 \
    --model "uie-base" \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device "gpu"
#evaluation
python evaluate.py \
    --model_path "./checkpoint/model_best" \
    --test_path "./data/test.txt" \
    --batch_size 16 \
    --max_seq_len 512