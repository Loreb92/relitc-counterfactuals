MAX_INPUT_LENGTH=256

source $1    # Shareed params
source $2    # Task and dataset args

CUDA_VISIBLE_DEVICES=0 python extract_feature_importances.py \
        --data-path ${INPUT_DATA_PATH} \
        --model-path ${INPUT_MODELS_PATH} \
        --output-path ${DATA_PATH} \
        --dataset ${dataset} \
        --max-input-length ${MAX_INPUT_LENGTH}\
        --explain-wrt ${explain_wrt} \
       | tee logs/log-feature_importance-${dataset}-${explain_wrt}.log