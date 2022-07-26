cd apex
pip install -v --no-cache-dir ./ > log_apex_2.txt 2>&1
cd ..
pip install --user transformers > log.txt 2>&1

#pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html > log.txt 2>&1
export CUDA_VISIBLE_DEVICES=2
#MODE=hybrid        # raw, contrastive, hybrid
#CONTRA_MASK=2      # 0,1,2
#QUEUE_SIZE=65536   # 65536, 16384, 4096, 131072
WEIGHT=0.5         # 0.25， 0.5, 1.0, 2.0
MODEL_TYPE=plbart
TEST=checkpoint-400-0.202/whole_model.bin
MODEL_NAME=uclanlp/plbart-base  # roberta-base, microsoft/codebert-base, microsoft/graphcodebert-base
MODEL_NAME_ALIAS=${MODEL_NAME/'/'/-}
#SAVED_PATH=../result/checkpoint-80000-0.321/
#SAVED_PATH=../result/checkpoint-50000-0.204/
#SAVED_PATH=../result/commitBart_WO_pos_plbart/checkpoint-60000-0.327/
#SAVED_PATH=../result/pretrain_None_plbart/checkpoint-96000-0.285/
#SAVED_PATH=../result/commitBart_WO_contra_plbart/checkpoint-80000-0.319/
#SAVED_PATH=../result/commitBart_WO_gtif_plbart/checkpoint-80000-0.323/
SAVED_PATH=../result/commitBart_WO_msg_plbart/checkpoint-60000-0.286/
#SAVED_PATH=../result/commitBart_WO_tif_plbart/checkpoint-50000-2.057/
IGNORE=None    #rename_var_names, rename_func_names, sample_funcs, insert_funcs, reorder_funcs, delete_token_docstrings, switch_token_docstrings, copy_token_docstring
FINE_TUNE=patch
#LANGUAGE=c,csharp,java,javascript,php,python,typescript
LANGUAGE=c
#OUTPUT=../result/pos_${LANGUAGE}_savedModel_${FROM_SAVE}${MODEL_NAME_ALIAS}
#OUTPUT=../result/trsfm_${LANGUAGE}_${FINE_TUNE}_${MODEL_NAME_ALIAS}
OUTPUT=../result/WO_msg_${FINE_TUNE}_${LANGUAGE}_${MODEL_NAME_ALIAS}
#OUTPUT=../result/pretrain_plbart
#OUTPUT=../result/pretrain_None_plbart
#TRAIN_FILE=../data/f_single_pos_data/
#EVAL_FILE=../data/f_single_pos_data/
#TRAIN_FILE=../data/finetune_data/
#EVAL_FILE=../data/test/
TRAIN_FILE=../data/patch_data/
EVAL_FILE=../data/patch_data/
NODE_INDEX=0 && echo NODE_INDEX: ${NODE_INDEX}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=1 && echo NUM_NODE: ${NUM_NODE}
mkdir -p ${OUTPUT}
BLOCK_SIZE=512 # sentence length
TRAIN_BATCH_SIZE=16 #12 #32 # per gpu batch
EVAL_BATCH_SIZE=16 #12 #32
ACCUMULATE_STEPS=2 #6
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
ADAM_EPS=1e-6
MAX_STEPS=4000
WARMUP_STEPS=400 # 0.1 of max steps
SAVE_STEPS=400  #
BEAM_SIZE=1
TEST_STEP=4200

CUDA_LAUNCH_BLOCKING=1 python run_patch.py\
    --output_dir=$OUTPUT \
    --finetune_task=$FINE_TUNE \
    --config_name=$MODEL_NAME \
    --model_type=$MODEL_TYPE \
    --model_name_or_path=$MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$EVAL_FILE \
    --block_size $BLOCK_SIZE \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATE_STEPS \
    --learning_rate $LEARNING_RATE \
    --node_index $NODE_INDEX \
    --gpu_per_node $PER_NODE_GPU \
    --weight_decay $WEIGHT_DECAY \
    --adam_epsilon $ADAM_EPS \
    --max_grad_norm 1.0 \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --seed 123456 \
    --lang $LANGUAGE \
    --weight $WEIGHT \
    --ignore_type $IGNORE \
    --beam_size $BEAM_SIZE \
    --saved_path $SAVED_PATH \
    --test_step $TEST_STEP \
    #--test_path $TEST

