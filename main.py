import os
import json
import yaml
import hashlib
import argparse
import re
import threading
import atexit
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import openai
import json_repair
from diskcache import Cache

# ==========================================
# 1. 基础数据结构 (Entities)
# ==========================================

@dataclass
class SubtitleProcessData:
    index: int
    original_text: str
    translated_text: str = ""

class ASRDataSeg:
    def __init__(self, text: str, start_time: str, end_time: str, index: int, translated_text: str = ""):
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.index = index
        self.translated_text = translated_text

class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        self.segments = sorted(segments, key=lambda x: x.index)

    @staticmethod
    def from_srt(file_path: str) -> "ASRData":
        # 使用 utf-8-sig 处理可能存在的 BOM
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read().strip()
        
        segments = []
        # 更加健壮的正则表达式，兼容多种换行情况
        pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\s*\n\d+|\s*$)', re.DOTALL)
        for match in pattern.finditer(content):
            idx = int(match.group(1))
            times = match.group(2).split(' --> ')
            text = match.group(3).strip()
            segments.append(ASRDataSeg(text, times[0], times[1], idx))
        
        if not segments:
            raise ValueError("无法解析 SRT 文件，请检查格式是否标准。")
        return ASRData(segments)

    def to_srt(self, output_path: str, bilingual: bool = False):
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in self.segments:
                f.write(f"{seg.index}\n{seg.start_time} --> {seg.end_time}\n")
                if bilingual:
                    f.write(f"{seg.text}\n{seg.translated_text}\n\n")
                else:
                    f.write(f"{seg.translated_text if seg.translated_text else seg.text}\n\n")

# ==========================================
# 2. 翻译器基类 (BaseTranslator)
# ==========================================

class BaseTranslator(ABC):
    def __init__(self, thread_num: int, batch_num: int, source_lang: str, target_lang: str, cache_dir: str):
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache = Cache(cache_dir)
        self.executor = ThreadPoolExecutor(max_workers=thread_num)
        self.is_running = True
        atexit.register(self.stop)

    def translate(self, asr_data: ASRData) -> ASRData:
        translate_data_list = [
            SubtitleProcessData(index=seg.index, original_text=seg.text)
            for seg in asr_data.segments
        ]
        
        chunks = [translate_data_list[i:i + self.batch_num] for i in range(0, len(translate_data_list), self.batch_num)]
        translated_map = {}
        
        print(f"[*] 总共 {len(translate_data_list)} 条字幕，分为 {len(chunks)} 批处理...")
        
        futures = {self.executor.submit(self._safe_translate_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(futures):
            try:
                result_chunk = future.result()
                for data in result_chunk:
                    translated_map[data.index] = data.translated_text
            except Exception as e:
                print(f"[-] 批处理执行失败: {e}")

        for seg in asr_data.segments:
            seg.translated_text = translated_map.get(seg.index, "")
        
        return asr_data

    def _safe_translate_chunk(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        # 生成缓存键：包含模型参数、源语言、目标语言和文本内容的哈希
        key_data = [asdict(d) for d in chunk]
        cache_key = f"{self.__class__.__name__}:{generate_cache_key(key_data)}:{self.source_lang}:{self.target_lang}"
        
        cached = self.cache.get(cache_key)
        if cached:
            return [SubtitleProcessData(**d) for d in cached]

        result = self._translate_chunk(chunk)
        self.cache.set(cache_key, [asdict(d) for d in result], expire=86400 * 7)
        return result

    @abstractmethod
    def _translate_chunk(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        pass

    def stop(self):
        self.is_running = False
        self.executor.shutdown(wait=False)

def generate_cache_key(data: Any) -> str:
    data_str = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

# ==========================================
# 3. LLM 翻译器实现 (LLMTranslator)
# ==========================================

class LLMTranslator(BaseTranslator):
    MAX_STEPS = 3

    def __init__(self, api_config: dict, prompts: dict, thread_num: int, batch_num: int, 
                source_lang: str, target_lang: str, cache_dir: str, is_reflect: bool = False):
        
        # 显式调用父类构造函数，修复 TypeError
        super().__init__(
            thread_num=thread_num, 
            batch_num=batch_num, 
            source_lang=source_lang, 
            target_lang=target_lang, 
            cache_dir=cache_dir
        )
        
        self.client = openai.OpenAI(api_key=api_config['api_key'], base_url=api_config['base_url'])
        self.model = api_config['model']
        self.prompts = prompts
        self.is_reflect = is_reflect

    def _get_prompt(self, ptype: str) -> str:
        content = self.prompts.get(ptype, "")
        return (content
                .replace("${source_language}", self.source_lang)
                .replace("${target_language}", self.target_lang)
                .replace("${custom_prompt}", ""))

    def _translate_chunk(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        subtitle_dict = {str(d.index): d.original_text for d in chunk}
        prompt_type = "reflect" if self.is_reflect else "standard"
        system_prompt = self._get_prompt(prompt_type)
        
        try:
            result_dict = self._agent_loop(system_prompt, subtitle_dict)
            
            for d in chunk:
                val = result_dict.get(str(d.index), d.original_text)
                # 处理反思模式下的嵌套结构
                if self.is_reflect and isinstance(val, dict):
                    d.translated_text = val.get('native_translation', str(val))
                else:
                    d.translated_text = str(val)
        except Exception as e:
            print(f"[-] 批处理纠错失败，尝试逐条翻译备份方案: {e}")
            return self._translate_single_fallback(chunk)
            
        return chunk

    def _agent_loop(self, system_prompt: str, subtitle_dict: Dict[str, str]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(subtitle_dict, ensure_ascii=False)}
        ]
        
        last_resp_dict = {}
        for step in range(self.MAX_STEPS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            # 使用 json_repair 修复 LLM 可能返回的非标准 JSON
            resp_dict = json_repair.loads(content)
            last_resp_dict = resp_dict
            
            is_valid, error_msg = self._validate_response(resp_dict, subtitle_dict)
            if is_valid:
                return resp_dict
            
            print(f"[!] 纠错循环 Step {step+1}: {error_msg}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Error: {error_msg}. Please fix and output the complete JSON object again."})
        
        return last_resp_dict

    def _validate_response(self, resp_dict: Any, origin_dict: Dict[str, str]) -> Tuple[bool, str]:
        if not isinstance(resp_dict, dict): 
            return False, "返回结果不是一个有效的 JSON 对象 (Dict)"
        
        origin_keys = set(origin_dict.keys())
        resp_keys = set(resp_dict.keys())
        
        if origin_keys != resp_keys:
            missing = origin_keys - resp_keys
            extra = resp_keys - origin_keys
            return False, f"Key 数量不匹配. 缺失: {missing}, 多余: {extra}"
        
        if self.is_reflect:
            for k, v in resp_dict.items():
                if not isinstance(v, dict) or 'native_translation' not in v:
                    return False, f"Key '{k}' 的格式不符合反思模式的要求(需包含 native_translation)"
        
        return True, ""

    def _translate_single_fallback(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        system_prompt = self._get_prompt("single")
        for d in chunk:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": d.original_text}
                    ],
                    temperature=0.3
                )
                d.translated_text = resp.choices[0].message.content.strip()
            except Exception:
                d.translated_text = d.original_text
        return chunk

# ==========================================
# 4. 默认 Prompt 模板
# ==========================================

DEFAULT_PROMPTS = {
    "standard": """You are a professional subtitle translator specializing in ${target_language}.
<guidelines>
- Translations must follow ${target_language} expression conventions, flow naturally.
- Strictly maintain one-to-one correspondence of subtitle numbering.
- Source language is ${source_language}.
</guidelines>
<output_format>
{ "index": "Translated text" }
</output_format>""",
    
    "single": "Translate the following ${source_language} text into ${target_language}. Output only the result.",
    
    "reflect": """You are a professional subtitle translator. Source: ${source_language}, Target: ${target_language}.
<instructions>
Stage 1: Initial Translation.
Stage 2: Machine Translation Detection & Deep Analysis (Structural rigidity, cultural mismatch).
Stage 3: Native-Quality Rewrite.
</instructions>
<output_format>
{
  "index": {
    "initial_translation": "...",
    "reflection": "...",
    "native_translation": "..."
  }
}
</output_format>"""
}

# ==========================================
# 5. CLI 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LLM Subtitle Translator--by liug")
    parser.add_argument("-i", "--input", required=True, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", help="输出 SRT 文件路径")
    parser.add_argument("-s", "--source", default="Japanese", help="源语言 (默认: Japanese)")
    parser.add_argument("-l", "--lang", default="简体中文", help="目标语言 (默认: 简体中文)")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--reflect", action="store_true", help="开启反思模式 (更高质量，更多 Token)")
    parser.add_argument("--only-result", action="store_true", help="只输出翻译结果，不保留原文对照")
    
    args = parser.parse_args()

    # 加载或创建配置文件
    if not os.path.exists(args.config):
        config = {
            "api": {
                "api_key": "YOUR_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o"
            },
            "settings": {
                "thread_num": 4,
                "batch_num": 10,
                "cache_dir": "./.cache_sub_trans"
            },
            "prompts": DEFAULT_PROMPTS
        }
        with open(args.config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        print(f"[*] 已生成默认配置文件 {args.config}，请修改 API Key 后运行。")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 读取字幕
    print(f"[*] 正在读取: {args.input}")
    try:
        asr_data = ASRData.from_srt(args.input)
    except Exception as e:
        print(f"[-] 读取失败: {e}")
        return

    # 初始化翻译器
    translator = LLMTranslator(
        api_config=config['api'],
        prompts=config.get('prompts', DEFAULT_PROMPTS),
        thread_num=config['settings'].get('thread_num', 4),
        batch_num=config['settings'].get('batch_num', 10),
        source_lang=args.source,
        target_lang=args.lang,
        cache_dir=config['settings'].get('cache_dir', './.cache_sub_trans'),
        is_reflect=args.reflect
    )

    print(f"[*] 任务开始: {args.source} -> {args.lang} (模式: {'反思' if args.reflect else '标准'})")
    
    try:
        translated_data = translator.translate(asr_data)
        
        # 保存结果
        output_file = args.output or args.input.replace(".srt", f".{args.lang}.srt")
        translated_data.to_srt(output_file, bilingual=args.only_result)
        print(f"[+] 翻译成功！保存至: {output_file}")
    except Exception as e:
        print(f"[-] 翻译过程中出现异常: {e}")
    finally:
        translator.stop()

if __name__ == "__main__":
    main()
