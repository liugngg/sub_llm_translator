# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-powered subtitle translation tool that translates SRT files via OpenAI-compatible Chat Completions or Responses APIs. Supports batch processing, local caching, bilingual output, and export to both SRT and ASS formats. The tool is optimized for Japanese-to-Chinese translation with automatic detection and retry of untranslated subtitles.

## Development Commands

### Setup
```powershell
uv sync
```

### Running the CLI
```powershell
# Basic translation (Japanese to Simplified Chinese by default)
python sub_trans.py -i input.srt -o output.srt

# Specify languages and API service
python sub_trans.py -i input.srt -c config.yaml -a openai -s Japanese -l 简体中文 -o output.srt

# Batch process directory with ASS bilingual output
python sub_trans.py -i .\subtitles -o .\translated --format ass --bilingual
```

### Testing
```powershell
# Run all tests in the project environment (discovery avoids the standard-library test package)
uv run python -m unittest discover -s test -p "test_*.py" -v

# Run a specific test
uv run python -m unittest discover -s test -p test_subtitles.py -k test_parse_utf8_japanese_srt_and_clean_lines -v

# Compile-check modules
python -m compileall sub_trans.py prompts.py
```

### Building Executables
```powershell
# Install build dependencies
uv sync --group build

# Build directory distribution (recommended)
uv run --with pyinstaller pyinstaller --clean --noconfirm --onedir --console --name sub_llm_trans sub_trans.py

# Build single executable (larger, slower startup)
uv run --with pyinstaller pyinstaller --clean --noconfirm --onefile --console --name sub_llm_trans sub_trans.py
```

## Architecture

### Translation Pipeline

1. **Parsing**: `ASRData.from_srt()` reads SRT files into `ASRDataSeg` objects with index, timestamps, and text
2. **Cleaning**: `srt_clean()` removes redundant punctuation, repeated content, and filler words using regex patterns (`BLANK_RE`, `REPEAT_CONTENT_RE`)
3. **Merging**: `srt_merge()` combines identical adjacent subtitles within `merge_interval` (default 0.7s), respecting `max_duration` cap (default 8s)
4. **Batching**: Segments are grouped into chunks of `batch_num` entries (default 10)
5. **Concurrent Translation**: `BaseTranslator` uses ThreadPoolExecutor to process chunks in parallel (default 2 threads)
6. **Caching**: Successful translations are cached for 7 days using `diskcache`; results containing censorship markers expire after 1 hour, and transient failures are not cached. Keys include source/target languages, batch content, service URL, model, API format, temperature, and rendered prompt. API keys are excluded.
7. **Untranslated Detection**: For Japanese-to-Chinese, `_looks_untranslated()` checks if translations retain excessive kana or match the source after normalization. Suspect entries are retried once; persistent failures are marked with `【原文】`
8. **Export**: `prepare_output()` applies final cleanup and merge passes to both original and translated tracks before writing SRT or ASS files

### Key Classes

- **ASRDataSeg**: Single subtitle entry (index, start_time, end_time, text, translated_text)
- **ASRData**: Collection of segments with parsing, cleaning, merging, and export logic
- **BaseTranslator**: Abstract translator with retry logic, caching, progress bar, and thread pool management
- **LLMTranslator**: Holds an ordered list of service endpoints (`api_configs`), each with its own client/model/format. Implements `_translate_chunk()` via OpenAI-compatible APIs with an agent loop for JSON correction (up to 3 steps)

### Translation Workflow

The `LLMTranslator._translate_chunk()` method:
1. Builds a JSON object mapping subtitle indexes to source text
2. Sends system prompt (from `prompts.py`) + JSON batch to LLM with `response_format=json_object`
3. Validates returned JSON structure and keys match input
4. For Japanese-to-Chinese: detects potentially untranslated entries using kana density and normalized text comparison
5. Retries suspect entries with explicit "translate into target language" instruction
6. Falls back to marking persistent failures as `【原文】` in output

The system uses a 3-step agent loop (`_agent_loop`) where the model can self-correct JSON syntax or missing keys by receiving error feedback as additional user messages.

### Error Handling Hierarchy

- **FatalTranslationError**: Authentication/NotFound errors that should not be retried
- **TranslationError**: Retriable errors (rate limits, timeouts, malformed JSON)
- **CensoredTranslationError**: Whole-request rejection triggers binary batch isolation; rejected single entries use `【云端审查】source`. Explicit refusals within valid JSON mark only the affected entries.
- **Chunk-level retry**: `MAX_CHUNK_RETRIES=3` means three total attempts, with 1-second and 2-second delays. Exhausted chunks use `【翻译失败】source`; authentication/resource errors propagate. The SDK may also retry HTTP requests internally.
- **Endpoint failover**: `_process_chunk_with_retry()` tries endpoints in order starting from a shared preferred index; `FatalTranslationError` and `CensoredTranslationError` move to the next configured service. A successful endpoint becomes the new shared starting point. Only when every endpoint rejects does the batch split.
- **Throttling**: `settings.request_interval` (default 1s, `0` = unlimited) is enforced by `BaseTranslator._throttle()` under a shared lock, so the interval applies across worker threads.
- **Agent loop**: Each chunk attempts JSON correction up to `MAX_STEPS` (3) times before giving up

### Configuration Structure

`config.yaml` supports multiple API providers under `apis` (each with `api_key`, `base_url`, `model`, optional `format`). The CLI selects the primary service via `-a` or `default_api` and passes the remaining services as ordered fallbacks. Runtime parameters live under `settings`:

**Core Settings:**
- `thread_num`: Number of concurrent translation threads (default: 2)
- `batch_num`: Subtitles per batch sent to LLM (default: 10)
- `cache_dir`: Local cache directory path (default: `./.cache_sub_trans`)
- `temperature`: LLM sampling temperature (default: 0.3, range: 0-2)
- `timeout`: API request timeout in seconds (default: 45)
- `request_interval`: Minimum interval between API calls in seconds; 0 = unlimited (default: 1)
- `merge_interval`: Maximum time gap for merging identical adjacent subtitles in seconds (default: 0.7)
- `max_duration`: Maximum duration cap for merged subtitles in seconds (default: 8)
- `filter_non_semantic_vocalisation`: Filter out pure vocalisation noise before translation (default: true)

**Fast Mode Settings:**
- `fast_mode`: Enable fast translation mode that drops untranslated entries instead of retrying (default: true)
- `fast_mode_retry_threshold`: In fast mode, if dropped entries exceed this count per batch, retry the entire batch once (default: 3)

**Interactive Settings:**
- `api_selection_timeout`: Seconds to wait for user API selection before auto-selecting default (default: 3)

**Advanced:**
- `replacements`: Dict of regex patterns to substitute during cleaning

### Fast Mode Behavior

When `fast_mode` is enabled:
1. Suspect translations (with excessive kana, untranslated text, or censorship) are dropped from output entirely
2. If the number of dropped entries in a batch exceeds `fast_mode_retry_threshold`, the entire batch is retranslated once
3. After retry, entries still deemed untranslated are dropped permanently
4. This mode trades completeness for speed by avoiding per-entry retry loops

### Interactive API Selection

At translation start, if no API is specified via `-a` flag:
1. The program lists all configured APIs with their models and base URLs
2. User can select by number (1-N) or name within `api_selection_timeout` seconds
3. If no input is received within the timeout, the `default_api` is automatically selected
4. The selected API becomes the primary endpoint; others serve as ordered fallbacks

### Prompt Templates

`prompts.py` exports the `standard` prompt template with placeholders `${source_language}` and `${target_language}`. The template instructs the model to:
- Preserve tone, intensity, names, and intentional foreign words
- Return exactly one translation per input index in JSON format
- Not refuse, summarize, or add commentary
- Keep Chinese lines within ~25 characters and preserve existing line breaks

### Testing Strategy

`test/test_subtitles.py` uses standard library `unittest` to verify:
- UTF-8 SRT parsing and regex-based cleaning rules
- Configured `replacements` dict application
- Merge logic for identical adjacent segments within `merge_interval` and `max_duration` constraints
- Millisecond rounding and timestamp conversion
- SRT/ASS export with bilingual and single-language modes
- ASS timestamp format conversion (`HH:MM:SS,mmm` → `H:MM:SS.cs`)

`test/test_translation.py` uses mocked API responses to cover refusal detection, partial result preservation, batch isolation, retries, fatal errors, cache isolation, request throttling, and endpoint failover. Tests do not contact providers; manual CLI verification is still needed for real API compatibility.

## Important Constraints

- **Python 3.12+** required (see `pyproject.toml`)
- **API base_url handling**: Program auto-appends `/v1` if not present; handles both root URLs and explicit `/v1` endpoints
- **Cache expiration**: Successful chunk translations cached for 7 days; expired entries auto-cleaned on startup
- **Output naming**: SRT defaults to `<basename>.<target_lang>.srt`; ASS defaults to `<basename>.ass`
- **Bilingual format**: SRT uses separate lines (`translated\noriginal`); ASS uses `\N` line break marker
- **Build artifacts**: Generated `config.yaml`, `.cache_sub_trans/`, `dist/`, and output subtitle files should never be committed

## Code Style Notes

- Use four-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module-level regex constants
- Preserve prompt placeholder syntax `${variable}` exactly when editing templates
- Keep CLI argparse help text synchronized with actual behavior
- UTF-8 text handling throughout (BOM-aware via `utf-8-sig` encoding)
