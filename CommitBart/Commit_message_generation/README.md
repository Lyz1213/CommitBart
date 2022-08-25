# Downstream generation tasks
We provide the code for reproducing the experiments on commit message generation
## Data preprocess
Given a commit is made up of several components. We defined several special sep token to denote each -> ''[MSG]'' for commit message, ''[FILE]'' for file path, ''[CODE]'' For code snippet, ''[POS]/[END]'' for a postive change statement, and ''[NEG]/[END]'' for a negative change statement.
Besides, we also use segment embedding to embed tokens in each segment in CommitBART-base -> 0: commit message, 1: postive statmens, 2: negative statements, 3: file path, 4: code context
## Fine-tune
```shell   
MODEL_NAME=uclanlp/plbart-base
MODEL_NAME_ALIAS=${MODEL_NAME/'/'/-}
SAVED_PATH=../ckpt/CommitBART-base
FINE_TUNE=msg # fine-tune task: msg->commit message generation, pos->updated code snippet generation, sp->positive code statements generation
LANGUAGE=python #c,csharp,java,javascript,php,python,typescript
OUTPUT=../result/CommitBART_${FINE_TUNE}_${LANGUAGE}_${MODEL_NAME_ALIAS}
TRAIN_FILE=../data/finetune_data/ #for msg and pos
EVAL_FILE=../data/test/
NODE_INDEX=0 && echo NODE_INDEX: ${NODE_INDEX}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=1 && echo NUM_NODE: ${NUM_NODE}
mkdir -p ${OUTPUT}
BLOCK_SIZE=512 # max input length
TRAIN_BATCH_SIZE=16 #
EVAL_BATCH_SIZE=32
ACCUMULATE_STEPS=2
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
ADAM_EPS=1e-6
MAX_STEPS=10000
WARMUP_STEPS=1000 # 0.1 of max steps
SAVE_STEPS=2000  
BEAM_SIZE=1
TEST_STEP=10000

CUDA_LAUNCH_BLOCKING=1 python run_finetune.py\
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
    --test_step $TEST_STEP


```
## Inference and evaluation
For evaluation the fine-tuned model, you can simply add the argument based on above script
```shell  
TEST=../result/CommitBART_msg_python_uclanlp/plbart/yourcheckpoint.bin
--test_path $TEST
```