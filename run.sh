  #!/usr/bin/env bash
  set -euo pipefail

  # -------- CONFIGURATION --------
  MODEL_PATH=[/path/to/model/model.keras]
  TOKENIZER_PATH=[/path/to/tokenizer/tokenizer.pkl]
  TEST_PATH=[ADD]
  RESULT_PATH=[ADD]
  BATCH_SIZE=128
  MAXLEN=[ADD]
  TEST_PATTERN="*.csv"
  VERBOSE="--verbose"

  mkdir -p "$RESULT_PATH"

  # -------- RUN ----------
  python3 -u e2e_pipeline.py \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --test_path "$TEST_PATH" \
    --result_path "$RESULT_PATH" \
    --batch_size "$BATCH_SIZE" \
    --maxlen "$MAXLEN" \
    --test_pattern "$TEST_PATTERN" \
    --metrics_only \
    ${VERBOSE:-}
