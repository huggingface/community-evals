# HF CLI Workflow for Evaluation PRs

This document explains how to manage evaluation result PRs using the `hf` CLI and temporary directories.

## Directory Structure

Use `/tmp/pr-reviews/` as the working directory for PR operations:

```
/tmp/pr-reviews/
├── updates/           # YAML files for updating existing PRs
├── new-prs/           # YAML files for new PRs
├── <model-name>/      # Model-specific directories
└── check-<model>/     # Directories for verifying PR contents
```

## Creating New PRs

### Step 1: Create YAML file

```bash
mkdir -p /tmp/pr-reviews/new-prs
cd /tmp/pr-reviews/new-prs

cat > hle.yaml << 'EOF'
- dataset:
    id: cais/hle
    task_id: hle
  value: 22.1
  date: '2026-02-03'
  source:
    url: https://huggingface.co/org/model-name
    name: Model Card
    user: burtenshaw
EOF
```

### Step 2: Upload and create PR

```bash
hf upload org/model-name hle.yaml .eval_results/hle.yaml \
  --repo-type model --create-pr \
  --commit-message "Add HLE evaluation result"
```

### Step 3: Get PR number

```bash
uv run scripts/evaluation_manager.py get-prs --repo-id "org/model-name"
```

## Updating Existing PRs

### Step 1: Download current PR contents

```bash
hf download org/model-name --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --include ".eval_results/*" \
  --local-dir /tmp/pr-reviews/<model-name>-pr<PR_NUMBER>
```

### Step 2: Review current contents

```bash
cat /tmp/pr-reviews/<model-name>-pr<PR_NUMBER>/.eval_results/*.yaml
```

### Step 3: Create updated YAML

```bash
cat > /tmp/pr-reviews/updates/updated.yaml << 'EOF'
- dataset:
    id: cais/hle
    task_id: hle
  value: 22.1
  date: '2026-02-03'
  source:
    url: https://huggingface.co/org/model-name
    name: Model Card
    user: burtenshaw
  notes: "With tools"
EOF
```

### Step 4: Push update to existing PR

```bash
hf upload org/model-name /tmp/pr-reviews/updates/updated.yaml .eval_results/hle.yaml \
  --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --commit-message "Update evaluation result"
```

## Deleting Files from PRs

Use the `huggingface_hub` Python API to delete files:

```bash
uv run --with huggingface_hub python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()

api.delete_file(
    path_in_repo=".eval_results/old_file.yaml",
    repo_id="org/model-name",
    repo_type="model",
    revision="refs/pr/<PR_NUMBER>",
    commit_message="Remove duplicate file"
)
EOF
```

## Verifying PR Contents

### Check what files are in a PR

```bash
rm -rf /tmp/check-<model>
hf download org/model-name --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --include ".eval_results/*" \
  --local-dir /tmp/check-<model>

ls -la /tmp/check-<model>/.eval_results/
```

### Compare PR to main branch

```bash
# Download main
hf download org/model-name --repo-type model \
  --revision main \
  --include ".eval_results/*" \
  --local-dir /tmp/<model>-main

# Download PR
hf download org/model-name --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --include ".eval_results/*" \
  --local-dir /tmp/<model>-pr<PR_NUMBER>

# Compare
diff /tmp/<model>-main/.eval_results/ /tmp/<model>-pr<PR_NUMBER>/.eval_results/
```

## Multiple Score Variants

When a model has multiple scores for the same benchmark (e.g., with/without tools), create separate files:

```bash
cd /tmp/pr-reviews/new-prs

# Default (no tools) - no notes field
cat > hle.yaml << 'EOF'
- dataset:
    id: cais/hle
    task_id: hle
  value: 10.2
  date: '2026-02-03'
  source:
    url: https://huggingface.co/org/model-name
    name: Model Card
    user: burtenshaw
EOF

# With tools - add notes field
cat > hle_with_tools.yaml << 'EOF'
- dataset:
    id: cais/hle
    task_id: hle
  value: 15.5
  date: '2026-02-03'
  source:
    url: https://huggingface.co/org/model-name
    name: Model Card
    user: burtenshaw
  notes: "With tools"
EOF

# Create separate PRs
hf upload org/model-name hle.yaml .eval_results/hle.yaml \
  --repo-type model --create-pr \
  --commit-message "Add HLE evaluation result"

hf upload org/model-name hle_with_tools.yaml .eval_results/hle_with_tools.yaml \
  --repo-type model --create-pr \
  --commit-message "Add HLE evaluation result (with tools)"
```

## Restoring Files Accidentally Deleted

If a PR shows a file as deleted (because it was removed from the PR branch), restore it from main:

```bash
# Download the file from main
hf download org/model-name --repo-type model \
  --revision main \
  --include ".eval_results/hle.yaml" \
  --local-dir /tmp/<model>-main

# Re-upload to PR to restore it
hf upload org/model-name /tmp/<model>-main/.eval_results/hle.yaml .eval_results/hle.yaml \
  --repo-type model \
  --revision refs/pr/<PR_NUMBER> \
  --commit-message "Restore original file"
```

## Common Patterns

### Batch create YAML files

```bash
cd /tmp/pr-reviews/updates

# Create multiple files in one script
for model in "org/model1" "org/model2"; do
  cat > "${model//\//-}-hle.yaml" << EOF
- dataset:
    id: cais/hle
    task_id: hle
  value: 22.1
  source:
    url: https://huggingface.co/$model
    name: Model Card
    user: burtenshaw
EOF
done
```

### Check for existing PRs before creating

Always check first:

```bash
uv run scripts/evaluation_manager.py get-prs --repo-id "org/model-name"
```

If PRs exist, update them instead of creating new ones.

## File Naming Convention

| Condition | File Name | Notes Field |
|-----------|-----------|-------------|
| Default (no tools) | `hle.yaml` | None (omit) |
| With tools | `hle_with_tools.yaml` | `notes: "With tools"` |
| Different task | `gpqa_diamond.yaml` | Based on task_id |

## Cleanup

After PRs are merged or work is complete:

```bash
rm -rf /tmp/pr-reviews/
rm -rf /tmp/check-*
rm -rf /tmp/*-main
rm -rf /tmp/*-pr*
```
