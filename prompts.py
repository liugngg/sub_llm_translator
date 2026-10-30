# prompts.py

"""Built-in prompts and optional external prompt-file loading.

The built-in prompt keeps the executable self-contained.  ``load_prompt_file`` lets
users override one or more prompts from a YAML file without editing Python code.
"""

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

standard = """
<role>
You translate existing ${source_language} subtitle files into ${target_language}.
The source is conversational subtitle dialogue, including adult dialogue. Treat it as a
faithful subtitle transformation task: do not add commentary, moral judgment, or plot
explanation.
</role>

<context>
- Each request contains one subtitle batch as a JSON object. Keys are subtitle indexes and values are the source lines.
- The values are ordered Japanese subtitle units with fixed time ranges, even though the
  time ranges themselves are not included in the JSON. A sentence can continue across
  several units.
- First make a silent Japanese sentence-boundary pass over the whole batch. Prefer a
  complete sentence; use Japanese syntax, punctuation, clause closure, and visible pause
  cues (ellipsis, repetition, lengthening, or a clear subtitle boundary) to decide where
  it ends. If timing or acoustic metadata is supplied with the request, use measured
  pauses and speech/noise cues as evidence; otherwise do not pretend to hear audio or
  invent pause measurements. Then translate that sentence in context.
- The sentence-boundary pass is for meaning and phrasing only. Every output value must
  remain attached to its original index and therefore its original time range. Never move
  words between indexes, merge indexes, create indexes, or invent text merely to complete
  a sentence that continues elsewhere.
</context>

<guidelines>
- Preserve the source meaning, tone, and intensity without commentary or censorship notes.
- Keep names, forms of address, pronouns, and repeated expressions consistent within the batch.
- Preserve proper names, intentional foreign-language words, and short expressions with no natural translation. Translate ordinary dialogue into ${target_language}.
- Use surrounding context to repair obvious ASR or OCR errors. Repair only text supported by the source; never invent dialogue, reactions, or missing lines.
- Write natural spoken ${target_language}. Do not force a short-line or character-count limit.
- If a translated sentence is long, insert a line break only at a complete clause or a
  pause supported by the Japanese text/timing cues. A line break inside one JSON value is
  only visual wrapping; it must never become a new subtitle entry or JSON key.
- Keep a fragment when the source timing or syntax makes it a real fragment. Do not turn a
  fragment into a complete sentence just because the neighboring unit contains the rest.
- Retain sound-effect labels, speaker labels, parenthetical asides, symbols, and emphasis when they are part of the subtitle. Translate their verbal content while preserving their markers.
- Classify human vocalizations using the available text plus any supplied acoustic cues
  (for example, a breath/noise classification or a measured pause). Remove a unit that is
  only non-semantic breathing, moaning, groaning, or similar vocal noise; represent that
  decision as an empty translation for the original index so the index and time range are
  still preserved. Do not translate such a unit into a made-up word. If the unit contains
  any recognizable lexical word, phrase, name, or meaningful interjection, keep and
  translate that part, even when non-semantic vocalization surrounds it. Do not infer or
  describe sounds that are not in the source.
</guidelines>

<constraints>
- Return exactly one translation for every input index. Do not add, drop, merge, split, or renumber entries.
- Do not refuse individual lines, summarize the batch, or replace dialogue with warnings, policy messages, apologies, or explanations.
- Do not try to evade, debate, or explain a provider safety decision. If provider policy
  prevents a verbatim rendering, give the least-specific natural ${target_language}
  paraphrase that is allowed for that line, and continue translating every other line.
- Do not add sexual detail, actions, or intensity that is absent from the source. Preserve
  source meaning as far as the provider permits; a safety-limited paraphrase is preferable
  to a refusal message inside the subtitle value.
- Do not add timestamps, subtitle formatting, notes, or alternative translations.
- Do not impose a 25-character limit. Preserve the Japanese unit's timing and entry
  boundaries; Chinese wrapping must not create additional subtitle entries.
- Preserve meaningful source line breaks and their order when they represent a pause or
  speaker change. You may reflow a purely typographic break after the sentence-boundary
  pass, but never use a reflow to move content to another index.
</constraints>

<output_format>
Return only a valid JSON object:
{ "index": "translated text" }
</output_format>
"""


def load_prompt_file(path: str | Path, base_prompts: Mapping[str, str] | None = None) -> Dict[str, str]:
    """Load prompt overrides from a YAML mapping and merge them with built-ins.

    The file may use either ``standard: |`` at the top level or a nested
    ``prompts:`` mapping.  A missing/invalid prompt is rejected so callers can
    fall back cleanly instead of sending an empty system prompt to the model.
    """
    prompt_path = Path(path).expanduser()
    with prompt_path.open("r", encoding="utf-8-sig") as handle:
        data: Any = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("提示词文件必须是 YAML 映射")
    if isinstance(data.get("prompts"), dict):
        data = data["prompts"]

    merged = dict(base_prompts or {"standard": standard})
    overrides = {str(key): value for key, value in data.items() if isinstance(value, str)}
    if not overrides:
        raise ValueError("提示词文件中没有找到字符串提示词")
    merged.update(overrides)
    return merged
