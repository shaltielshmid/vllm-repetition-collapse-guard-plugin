# vLLM Repetition Collapse Guard Plugin

vLLM-PLugin that detects and stops LLM generation when it gets stuck in repetition collapse (short repeating loops of tokens or n-grams until max length).

## Installation

```bash
pip install "git+https://github.com/shaltielshmid/vllm-repetition-collapse-guard-plugin@main#egg=vllm-repguard"
```

## Usage

Enable the plugin by setting the environment variable:

```bash
VLLM_REPGUARD_ENABLE=1 vllm serve <model_name>
```

### Docker

```bash
docker run -e VLLM_REPGUARD_ENABLE=1 ... vllm/vllm-openai \
  bash -c "pip install 'git+https://github.com/shaltielshmid/vllm-repetition-collapse-guard-plugin@main#egg=vllm-repguard' && vllm serve <model_name>"
```

## Configuration

All parameters are configured via environment variables:

- **`VLLM_REPGUARD_ENABLE`** (default: `1`): Set to `0` or `false` to disable the plugin.

- **`BUFFER_SIZE`** (default: `1024`): Size of the token history buffer (must be a power of 2). Larger values detect longer-range patterns but use more memory per request.

- **`MAX_TOKEN_REP`** (default: `32`): Maximum consecutive repetitions of tokens before stopping. For single tokens - X occurences of the same token causes a hit. For n-grams, it's (at least) X // n repeats. 
Lower values catch single-token loops faster. 

- **`MIN_GRAM_REP`** (default: `5`): Minimum number of n-gram repetitions required to trigger a stop (works together with `MAX_TOKEN_REP`, the higher of the two for each n). 

- **`MAX_PERIOD_CHECK`** (default: `256`): Maximum distance to check for repeating patterns. Controls how far back the algorithm looks for pattern matches. Higher values detect longer-range repetitions but may impact performance.

Enjoy!
