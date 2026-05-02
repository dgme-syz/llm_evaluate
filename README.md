# llm_evaluate

A lightweight LLM evaluation framework for **machine translation** and **math reasoning**, built on Hydra + vLLM + Ray. It supports multi-dataset / multi-metric evaluation in both **offline** (local vLLM) and **online** (remote OpenAI-compatible API) modes.

> The math reasoning track is still under development. This document focuses on **translation evaluation**.

---

## Project Layout

```
llm_evaluate/
├── llm_evaluate/
│   ├── main_eval.py            # Hydra entry point
│   ├── config/
│   │   ├── config.yaml
│   │   ├── config_mt.yaml             # offline (vLLM) translation config
│   │   ├── config_mt_server.yaml      # online (OpenAI-compatible API) translation config
│   │   └── data/
│   │       ├── translation/           # translation dataset yamls
│   │       └── math/                  # math dataset yamls (WIP)
│   ├── dataset/
│   │   ├── abstract.py                # EvalDataset abstract base
│   │   ├── registry.py                # @register decorator
│   │   ├── translation/
│   │   │   ├── __init__.py            # language code tables + prompt templates
│   │   │   ├── flores.py
│   │   │   ├── wmt24.py
│   │   │   └── challenge_set.py
│   │   └── math/                      # WIP
│   ├── utils/
│   │   ├── metric/                    # BLEU / COMET / BLEURT / xCOMET / ...
│   │   └── eval_func/                 # base / recheck / no_steps / split / ...
│   ├── vllm/                          # local vLLM model builder
│   └── worker/                        # Ray-managed GPU workers
├── scripts/                           # entry shell scripts
│   ├── run_batch_mt.sh                # offline batch runner
│   ├── run_batch_mt_server.sh         # online (API) batch runner
│   ├── run_mt.sh                      # single-model offline launcher
│   └── run_mt_server.sh               # single-model online launcher
└── pyproject.toml
```

Call chain: `main_eval.py` loads the top-level config → resolves the `data` list (each item points to a yaml under `config/data/...`) → fetches the registered dataset class via `get_dataset(name)` → the class formats prompts with the templates from [translation/\_\_init\_\_.py](llm_evaluate/dataset/translation/__init__.py) → vLLM (offline) or an OpenAI-compatible client (online) generates → the metrics listed under `evaluate.metrics` produce scores.

---

## Quick Start

Installation:

```bash
pip install -e .
```

Translation evaluation always starts from one of the two batch scripts:

| Mode | Backend | Config | Batch script |
| --- | --- | --- | --- |
| Offline | local vLLM | [config_mt.yaml](llm_evaluate/config/config_mt.yaml) | [scripts/run_batch_mt.sh](scripts/run_batch_mt.sh) |
| Online | remote OpenAI-compatible API | [config_mt_server.yaml](llm_evaluate/config/config_mt_server.yaml) | [scripts/run_batch_mt_server.sh](scripts/run_batch_mt_server.sh) |

```bash
# Offline (local checkpoints over vLLM)
bash scripts/run_batch_mt.sh

# Online (call a hosted API such as DeepSeek / Qwen-MT / GPT / Gemini ...)
bash scripts/run_batch_mt_server.sh
```

Outputs are written as jsonl under `outputs_dir / exp_tag (or server.model) / data_tag` defined in the active config.

### Data sources

The bundled translation yamls under [llm_evaluate/config/data/translation/](llm_evaluate/config/data/translation/) point at the following upstream datasets — each `data_path` is either a Hugging Face repo id (resolved via `datasets.load_dataset`) or a local cache thereof:

| Dataset | `data_path` |
| --- | --- |
| FLORES (BenchMAX general translation) | [`LLaMAX/BenchMAX_General_Translation`](https://huggingface.co/datasets/LLaMAX/BenchMAX_General_Translation) |
| WMT24 | download from [box.nju.edu.cn/d/6de5478fea1544f2b3f4](https://box.nju.edu.cn/d/6de5478fea1544f2b3f4/), then point `data_path` at the extracted folder |
| MT mixed train (WMT 17–20) | download from [box.nju.edu.cn/d/754ee94e447f453dbc35](https://box.nju.edu.cn/d/754ee94e447f453dbc35/), then point `data_path` at the extracted folder |
| Seed-X challenge set | download from [box.nju.edu.cn/d/d9dc4055cd2743368ac4](https://box.nju.edu.cn/d/d9dc4055cd2743368ac4/), then point `data_path` at the extracted folder (upstream: [Seed-X-7B](https://github.com/ByteDance-Seed/Seed-X-7B/tree/main/challenge_set)) |
| FLORES-200 dev (debug) | [`DGME/FLORES-200`](https://huggingface.co/datasets/DGME/FLORES-200) |

If you keep a local mirror, just point `data_path` at the on-disk directory; otherwise the HF repo id will be downloaded on first use.

### What the batch scripts do

Both scripts share the same skeleton: declare a Bash associative array `models`, where each entry is `<model_id> = "<model_path_or_name> <hydra overrides...>"`. The script iterates over every enabled entry and forwards it to the per-model launcher (`run_mt.sh` or `run_mt_server.sh`), which in turn invokes:

```
python -m llm_evaluate.main_eval --config-name <config_mt | config_mt_server> <hydra overrides>
```

To run a new model you only need to add (or uncomment) one line in the array. Anything after the first whitespace-separated token is appended as Hydra overrides on top of the base config.

**[run_batch_mt.sh](scripts/run_batch_mt.sh)** — offline runner. Each entry looks like:

```bash
models["Qwen2.5-7B-Instruct"]="/path/to/Qwen2.5-7B-Instruct \
    generate_params.offline.sampling_params.temperature=0.6 \
    ++generate_params.offline.sampling_params.top_p=0.95 \
    ++generate_params.offline.sampling_params.top_k=20 \
    ++llm.prompt_template=qwen_chat_mt \
    ++tokenize_args.enable_thinking=false \
    ++evaluate.eval_func=base \
    ++llm.exp_tag=Qwen2.5-7B-Instruct"
```

The script also adds:

- `MAX_RETRIES=3` retry loop per model.
- A free-GPU detector inside `run_mt.sh` that waits until enough idle GPUs are available, then sets `CUDA_VISIBLE_DEVICES` and `llm.vllm.tensor_parallel_size` automatically (1 / 2 / 4).
- A V100 branch that switches to `VLLM_ATTENTION_BACKEND=XFORMERS`, `dtype=float32`, and disables chunked prefill.
- Per-model logs under `./logs/main/`.

You typically only edit:
1. The `models` map (model path + sampling / template / tag overrides).
2. The `data` list inside [config_mt.yaml](llm_evaluate/config/config_mt.yaml) to choose which datasets to run.

**[run_batch_mt_server.sh](scripts/run_batch_mt_server.sh)** — online runner. Each entry binds the model name and API credentials in one line:

```bash
models["deepseek-v3.2"]="'deepseek-v3.2' \
    server.openai_args.api_key=sk-xxxx \
    server.openai_args.base_url='https://api.bltcy.ai/v1/' \
    ++server.prompt_template=qwen_chat_mt"
```

The first token becomes `server.model`; the remaining tokens override fields in [config_mt_server.yaml](llm_evaluate/config/config_mt_server.yaml) (`server.openai_args.api_key`, `server.openai_args.base_url`, `server.prompt_template`, …). `use_server: true` in that config flips `main_eval.py` from local vLLM to an OpenAI-compatible client.

You typically only edit:
1. The `models` map (model id + `api_key` + `base_url` + prompt template).
2. The `data` list and `evaluate.metrics` inside [config_mt_server.yaml](llm_evaluate/config/config_mt_server.yaml).

---

## Developing a New Translation Dataset

Only add a new dataset class when an existing class cannot parse your data layout. Otherwise just write a new yaml under [config/data/translation/](llm_evaluate/config/data/translation/) and reuse `flores` / `wmt24` / `challenge_set`.

### 1. Register a dataset class

Create `your_dataset.py` under [llm_evaluate/dataset/translation/](llm_evaluate/dataset/translation/), subclass `EvalDataset`, and register it with `@register("your_name")`:

```python
from llm_evaluate.dataset import EvalDataset, register
from llm_evaluate.dataset.translation import get_prompt_template, resolve_lang_name


@register("your_dataset")
class YourDataset(EvalDataset):
    def __init__(self, data_dir, subset_name=None, split="train", builder=None, extra_args=None):
        super().__init__(data_dir, subset_name, split, builder)

        self.extra_args = extra_args or {}
        if "prompt_template" not in self.extra_args:
            raise ValueError("your_dataset requires `prompt_template` in extra_args.")
        if not isinstance(subset_name, str) or "-" not in subset_name:
            raise ValueError(f"your_dataset expects subset_name like 'en-zh_CN', got {subset_name!r}.")

        src_code, tgt_code = subset_name.strip().split("-", 1)
        self.src_code = src_code
        self.tgt_code = tgt_code.split("_")[0]
        # `resolve_lang_name` first tries LANGUAGE_BY_CODE (locale-aware), then
        # falls back to LANG_DICT on the bare code, then to the raw code itself.
        self.src_lang = resolve_lang_name(src_code)
        self.tgt_lang = resolve_lang_name(tgt_code)
        self.template_func = get_prompt_template(self.extra_args["prompt_template"])

    def convert_item(self, examples, **kwargs) -> dict:
        src_text = examples["source"]
        tgt_text = examples["target"]

        return {
            "data_source": "your_dataset",
            "prompt": self.template_func(self.src_lang, self.tgt_lang, src_text),
            "ability": "translation",
            "reward_model": {"ground_truth": tgt_text, "style": "rule"},
            "extra_info": {
                "src": src_text,
                "src_lang": self.src_code,
                "tgt_lang": self.tgt_code,
            },
        }
```

The class is responsible for:

- **Loading the raw data.** Forward `data_path / subset_name / split / builder` to the base class — `EvalDataset` calls `datasets.load_dataset` and supports both single-source and parallel multi-source modes (e.g. one file for source, one for target).
- **Resolving languages.** Read source / target codes from `subset_name` and translate them with [`resolve_lang_name`](llm_evaluate/dataset/translation/__init__.py) into the readable names that prompt templates expect. The helper layers three lookups in order: `LANGUAGE_BY_CODE` (locale codes such as `zh_CN` → `"Chinese (Simplified)"`), `LANG_DICT` on the bare language code (e.g. `en_XX` → `en` → `"English"`), and finally the raw input as a last-resort fallback. The original tables and `LANGUAGE_PAIRS` (the supported pair allow-list) remain available if you need them directly.
- **Building the prompt.** Use `get_prompt_template(name)` to fetch one of the templates registered in [translation/\_\_init\_\_.py](llm_evaluate/dataset/translation/__init__.py):

  | Template | Target model |
  | --- | --- |
  | `qwen_chat_mt` | Generic Qwen chat translation |
  | `qwen_think_mt` | Qwen with thinking |
  | `qwen_mt_turbo` | Qwen-MT |
  | `hunyuan_mt_chat` | Hunyuan-MT |
  | `seed_x_mt` | Seed-X PPO |
  | `ssr_x_mt` | SSR-X-Zero |
  | `tower_chat_input` / `tower_plus_input` | Tower / Tower+ |

  Add new templates by appending to the `PROMPT_TEMPLATE` `OrderedDict` at the bottom of `__init__.py`, or call `register_prompt_template(name, func, overwrite=False)` from your own module so the registration is co-located with the function definition.
- **Implementing `convert_item`.** Map each raw row to the unified record: `prompt` (chat messages), `reward_model.ground_truth` (reference translation) and `extra_info` (the source text, source / target language tags, and anything else metrics may need).

Finally, import the class in [llm_evaluate/dataset/\_\_init\_\_.py](llm_evaluate/dataset/__init__.py) so the `@register` decorator runs at import time. All three registries (`dataset`, `metric`, `eval_func`) accept an `overwrite=True` flag for cases where you intentionally want to replace an existing entry — the default behavior is to fail loudly on duplicates.

### 2. Write a dataset yaml

Add a yaml under [llm_evaluate/config/data/translation/](llm_evaluate/config/data/translation/), modeled on [flores_en2zh.yaml](llm_evaluate/config/data/translation/flores_en2zh.yaml):

```yaml
name: your_dataset                # must match @register
data_path:                        # one path or several (parallel files)
  - /path/to/source_data
  - /path/to/target_data
subset_name:                      # subset / language pair, passed to the dataset class
  - your_en
  - your_zh
split:                            # same length as data_path
  - train
  - train
num_samples: null                 # set to N for a quick smoke test
generate_args:                    # optional: per-dataset sampling overrides
  translation_options:
    source_lang: auto
    target_lang: Chinese
dataset_args:                     # forwarded as extra_args to the dataset class
  src_key: source
  tgt_key: target
data_tag: "your_dataset"          # used in output filenames
save: true                        # cache the processed dataset under temp/
```

### 3. Wire it into the master config

Append the yaml (path relative to `config/`, no `.yaml` suffix) to the `data` list in either [config_mt.yaml](llm_evaluate/config/config_mt.yaml) (offline) or [config_mt_server.yaml](llm_evaluate/config/config_mt_server.yaml) (online):

```yaml
data:
  - data/translation/your_dataset_en2zh

evaluate:
  val_size: 4096                  # batch size for inference
  metrics:                        # toggle as needed
    - BLEU
    - comet-22
    - cometkiwi
    - BLEURT
  eval_func: recheck              # base / recheck / no_steps / split / ...
  eval_func_args:
    check_num: 1
    use_thinking: false
  merge_strategy: total           # how multi-sample outputs are combined

# offline only
llm:
  prompt_template: qwen_chat_mt
  exp_tag: default
  vllm:
    model: /path/to/your/model
    tokenizer: /path/to/your/model
    dtype: bfloat16
    gpu_memory_utilization: 0.7
    tensor_parallel_size: 4

# online only
use_server: true
server:
  model: deepseek-v3.2
  prompt_template: qwen_chat_mt
  openai_args:
    api_key: ${oc.env:DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com/v1
```

Then add a row to the relevant batch script and run it.

---

## Math Reasoning (WIP)

`llm_evaluate/dataset/math/` already contains skeletons for `gsm8k` / `math500` / `aime2024` / `aime2025` / `llm_judge`, with matching metrics in [utils/metric/math.py](llm_evaluate/utils/metric/math.py). The integration shape mirrors translation (`@register` + `config/data/math/*.yaml` + master config), but the interfaces and evaluation flow are still in flux — not recommended for production use yet.

---

## License

Apache-2.0
