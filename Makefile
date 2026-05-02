.PHONY: setup eval

# One-time setup: init the eval submodule and install all requirements.
setup:
	git submodule update --init --recursive
	pip install -e .
	pip install -r third_party/babylm-eval/strict/requirements.txt

# Evaluate a checkpoint on the strict-small track.
# Usage: make eval MODEL=<hf_id_or_local_path> [BACKEND=mlm]
BACKEND ?= mlm
eval:
	@test -n "$(MODEL)" || (echo "Usage: make eval MODEL=<hf_id_or_path> [BACKEND=mlm]" && exit 1)
	bash scripts/evaluate.sh $(MODEL) $(BACKEND)
