export PYTHONPATH="."

device=1

python src/run.py \
    --model_name CRec \
    --device $device \
    --dataset_name mimic3 \
    --epochs 50 \
    --w_pos 0.1 \
    --w_neg 0.5 \
    --w_reg 0.1
