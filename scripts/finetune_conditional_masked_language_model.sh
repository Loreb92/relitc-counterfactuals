source $1    # Shareed params
source $2    # Task and dataset args
source $3    # finetuning args 

CUDA_VISIBLE_DEVICES=0 python finetune_conditional_masked_language_model.py \
        --data-path ${DATA_PATH} \
        --model-path ${INPUT_MODELS_PATH} \
        --model-output-dir ${FT_CMLM_PATH} \
        --model-name bert-base-uncased \
        --dataset ${dataset} \
        --explain-wrt ${explain_wrt} \
        --min-mask-frac ${MIN_MASK_FRAC} \
        --max-mask-frac ${MAX_MASK_FRAC} \
        --seed ${seed} \
        --learning-rate ${LEARNING_RATE} \
        --train-batch-size ${TRAIN_BATCH_SIZE} \
        --eval-batch-size ${EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${ACCUMULATION_STEPS} \
        --train-epochs ${TRAIN_EPOCHS} \
        --warmup-steps ${WARMUP_STEPS} \
        --lr-scheduler ${LR_SCHEDUDLER} \
        --patience ${PATIENCE} \
        --logging-strategy ${LOGGING_STRATEGY} \
        --logging-steps ${LOGGING_STEPS} \
        --save-strategy ${SAVE_STRATEGY} \
        --save-total-limit ${SAVE_TOTAL_LIMIT} \
        --evaluation-strategy ${EVAL_STRATEGY} \
        --load-best-model-at-end ${LOAD_BEST_MODEL_AT_END} \
        | tee logs/log-finetuning_CMLM-${dataset}-${explain_wrt}-${MIN_MASK_FRAC}-${MAX_MASK_FRAC}.log
        