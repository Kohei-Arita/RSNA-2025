# make train / make infer / make submit / make deps-freeze I

# ===== Kaggle オフライン提出補助タスク =====

# 前計算→Dataset用梱包（スケルトン）
kaggle-prep:
	python tools/pack_precompute.py

# 乾式リハーサル（ローカル）
# WORK_DIR/INPUT_DIR をローカルに向ければ動作確認可能
kaggle-dryrun:
	WORK_DIR=$(PWD)/.work INPUT_DIR=$(PWD)/.input python kaggle/kaggle_infer.py || true
	python tools/verify_submission.py .work/submission.csv || true

# wheels 収集（必要時のみ）
wheels:
	mkdir -p dist/wheels
	pip download -r kaggle/offline_requirements.txt -d dist/wheels || true