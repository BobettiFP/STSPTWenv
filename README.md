# STSPTWenv

Stochastic TSP with Time Windows (STSPTW) environment and POMO+PIP training/evaluation. All commands below are run from the **`POMO+PIP/`** directory.

```bash
cd POMO+PIP
```

---

## Training

Entry point: **`train.py`**

**Problems:** `--problem` one of `TSPTW`, `STSPTW`.

**Model types:** `--model_type` one of `POMO`, `POMO_STAR`, `POMO_STAR_PIP` (POMO only / + Lagrangian / + PIP masking).

**Example – TSPTW n=10, hard, POMO_STAR:**

```bash
python train.py --problem TSPTW --problem_size 10 --pomo_size 10 --hardness hard --model_type POMO_STAR \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 --validation_interval 500 --model_save_interval 1000
```

**Example – STSPTW n=10, pre-decision noise, POMO_STAR_PIP:**

```bash
python train.py --problem STSPTW --problem_size 10 --pomo_size 10 --hardness hard --model_type POMO_STAR_PIP \
  --reveal_delay_before_action --delay_scale 0.1 \
  --epochs 10000 --train_episodes 10000 --train_batch_size 1024 \
  --val_episodes 10000 --validation_batch_size 1000 --validation_interval 500 --model_save_interval 1000
```

**STSPTW-only flags:**

| Flag | Meaning |
|------|--------|
| `--delay_scale` | Delay magnitude (default `0.1`) |
| `--reveal_delay_before_action` | Pre-decision noise: sample travel times before action; omit for post-decision |

Checkpoints and logs go under `results/<run_name>/` (run name includes problem, size, hardness, `_dw...` for STSPTW, `_LM` for timeout reward, `_PIMask_*Step` for PIP).

---

## Evaluation

Entry point: **`test.py`**

**Example – evaluate a checkpoint on STSPTW n=10 hard:**

```bash
python test.py --problem STSPTW --problem_size 10 --hardness hard --pomo_size 1 \
  --checkpoint results/<run_name>/epoch-10000.pt \
  --test_set_path ../data/TSPTW/tsptw10_hard.pkl \
  --test_set_opt_sol_path ../data/TSPTW/lkh_tsptw10_hard.pkl \
  --aug_factor 8 --generate_PI_mask --pip_step 1
```

If `--test_set_path` and `--test_set_opt_sol_path` are omitted, defaults under `../data/<problem>/` are used (for STSPTW the data problem is TSPTW, e.g. `tsptw10_hard.pkl`).

**STSPTW evaluation with pre-decision noise (match training):**

```bash
python test.py --problem STSPTW --problem_size 10 --hardness hard \
  --checkpoint results/<run_name>/epoch-10000.pt \
  --reveal_delay_before_action --delay_scale 0.1 \
  --aug_factor 8 --generate_PI_mask --pip_step 1
```

---

## Summary

- **Working directory:** `POMO+PIP/`
- **Train:** `python train.py --problem <TSPTW|STSPTW> --problem_size N --pomo_size N --hardness <easy|medium|hard> --model_type <POMO|POMO_STAR|POMO_STAR_PIP> [--reveal_delay_before_action] [other args]`
- **Test:** `python test.py --problem ... --checkpoint <path> [--reveal_delay_before_action] [other args]`

Shell scripts (e.g. `run_train_n10.sh`) are not tracked; use the commands above as the reference.
