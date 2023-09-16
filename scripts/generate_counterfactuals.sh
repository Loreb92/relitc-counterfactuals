source $1    # Shareed params
source $2    # Task and dataset args
source $3    # generation args 

if [ -z "$6" ]; then
  echo >&2 'Missing parameter baseline_editor, setting to False'
  BASELINE_EDITOR="False"
  else
  echo >&2 'Settting baseline editor to True'
  BASELINE_EDITOR="True"
fi

CUDA_VISIBLE_DEVICES=0 python generate_counterfactuals.py \
        --data-path ${DATA_PATH} \
        --dataset ${dataset} \
        --explain-wrt ${explain_wrt} \
        --seed ${seed} \
        --min-mask-frac ${MIN_MASK_FRAC} \
        --max-mask-frac ${MAX_MASK_FRAC} \
        --results-folder ${RESULTS_PATH} \
        --editor-model-path ${FT_CMLM_PATH}\
        --blackbox-model-path ${INPUT_MODELS_PATH}\
        --sampling true \
        --top-p ${TOPP} \
        --top-k ${TOPK} \
        --max-search-levels ${MAX_SEARCH_LEVELS} \
        --n-edited-texts ${N_EDITED_TEXTS} \
        --next-token-strategy $4 \
        --direction $5\
        --fluency-model-path ${FLUENCY_MODEL_PATH}\
        --sem-similarity-model-path ${SEM_SIMILARITY_MODEL_PATH}\
        --baseline-editor ${BASELINE_EDITOR} \
        | tee logs/log-generation_CMLM-${dataset}-${explain_wrt}-${MIN_MASK_FRAC}-${MAX_MASK_FRAC}-$4-$5-${BASELINE_EDITOR}.log