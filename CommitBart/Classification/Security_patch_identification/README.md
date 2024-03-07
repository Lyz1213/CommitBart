# CommitBART
CommitBART is a Encoder-Decoder pre-trained model on GitHub commit. The model covers 7 programming languges (C, CSharp, JAVA, JAVAScript, PHP, Python, TypeScript). 

# Benchmark
We proposed a benchmark for research on commit-related task. We collect 7M instances of commits from top-ranked GitHub projects for pre-training, whereas the other 500K data is used for fine-tuning. The data can be found at https://drive.google.com/file/d/1sXYZeP-hwTrwTwa_RQF4qLOvAPEqjNRI/view?usp=sharing. Moreoer the Security Patch Identification dataset could be found at https://drive.google.com/file/d/186RzNb8uGHM5Wwx28DklBEPjUZCUjXhY/view?usp=sharing.

##Fine-tune
Your could directly run the run_patch.sh for fine-tuning the model for security patch detection.
Or run the following script:
``` shell
MODEL_TYPE=plbart
TEST=checkpoint-400-0.202/whole_model.bin #Your trained model ckpt for test
MODEL_NAME=uclanlp/plbart-base  # roberta-base, microsoft/codebert-base, microsoft/graphcodebert-base
MODEL_NAME_ALIAS=${MODEL_NAME/'/'/-}
SAVED_PATH=./result/CommitBART-base/ #Your model path
IGNORE=None    #rename_var_names, rename_func_names, sample_funcs, insert_funcs, reorder_funcs, delete_token_docstrings, switch_token_docstrings, copy_token_docstring
FINE_TUNE=patch
LANGUAGE=c
OUTPUT=../result/patch_${FINE_TUNE}_${LANGUAGE}_${MODEL_NAME_ALIAS}
TRAIN_FILE=./data/patch_data/ #Your data path
EVAL_FILE=./data/patch_data/
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
    --beam_size $BEAM_SIZE \
    --saved_path $SAVED_PATH \
    --test_step $TEST_STEP \
    #--test_path $TEST
```

##Evaluation
For evaluation of the fine-tuned modeo, you can simply add the argument based on the above script:

```shell  
TEST=#The path of the ckpt of the tuned model
--test_path $TEST
```