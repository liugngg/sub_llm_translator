import os
import json
import yaml
import hashlib
import argparse
import re
import time
import atexit
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import openai
import json_repair
from diskcache import Cache
from tqdm import tqdm  # 导入 tqdm

# 导入外部定义的 Prompts
import prompts

# 默认ASS文件头
ass_header = """[Script Info]
Title: Default Aegisub file
ScriptType: v4.00+
WrapStyle: 0
PlayResX: 1280
PlayResY: 960
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Microsoft YaHei,60,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# ###########需要清理的内容#######################
# 去掉开头的标点符号和空白符
BLANK_HEAD_RE = re.compile(r'^[^\w(（\[「【\'"‘“-]+', flags=re.UNICODE|re.MULTILINE)
# text = re.sub(r'^[\W]+', '', text, flags=re.UNICODE)

# 将文本中重复了2次及以上的多字字符串替换为1次
REPEAT_CONTENT_RE = re.compile(r'(.{2,})([\s,.!?;，。！？；]+)\1', flags=re.UNICODE|re.MULTILINE)

# 如果一行完全由语气词（呃 / 诶 / 啊…，但‘嗯’则保留）或标点组成，则替换为空
BLANK_RE = re.compile(r'^[ ,.，。！、!?？：；;—\-\–…\"''~「」『』啊嗬嗯哈唔哎呼咿呜呀西咻昂呐恩库莫伊阿咕哒喽呗嘛哟哇呃哦啦唉欸诶喔哼嘿喂干燥咚哔�んっはいふぁっあうちゅちあたえ]*$', flags=re.UNICODE|re.MULTILINE)
###############################################################

# ==========================================
# 1. 基础数据结构 (Entities)
# ==========================================

@dataclass
class SubtitleProcessData:
    index: int
    original_text: str
    translated_text: str = ""

# ASRDataSeg and ASRData 类保持逻辑不变，仅确保 clean_line 等方法正常工作
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
    def clean_line(text: str) -> str:
        text = text.strip()
        if not text: return ''
        # 将文本中重复了2次及以上的多字字符串替换为1次
        text = REPEAT_CONTENT_RE.sub(r'\1', text)
        # 去掉开头的标点符号和空白符
        text = BLANK_HEAD_RE.sub(r'', text)
        # 如果一行完全由语气词（呃 / 诶 / 啊…，但‘嗯’则保留）或标点组成，则替换为空
        text = BLANK_RE.sub(r'', text)
        return text.strip()

    @staticmethod
    def from_srt(file_path: str) -> "ASRData":
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read().strip()
        segments = []
        pattern = re.compile(r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\s*\n\d+|\s*$)', re.DOTALL)
        for match in pattern.finditer(content):
            idx = int(match.group(1))
            times = match.group(2).split(' --> ')
            text = ASRData.clean_line(match.group(3))
            if text:
                segments.append(ASRDataSeg(text, times[0], times[1], idx))
        return ASRData(segments)

    def _srt_to_ass_time(self, srt_time: str) -> str:
        t = srt_time.replace(',', '.')
        if t.startswith('0'): t = t[1:]
        parts = t.split('.')
        if len(parts) > 1:
            ms = parts[1][:2]
            t = f"{parts[0]}.{ms}"
        return t

    def to_srt(self, output_path: str, bilingual: bool = False):
        with open(output_path, 'w', encoding='utf-8') as f:
            for seg in self.segments:
                f.write(f"{seg.index}\n{seg.start_time} --> {seg.end_time}\n")
                src = ASRData.clean_line(seg.text)
                translated = ASRData.clean_line(seg.translated_text)
                if not translated: continue
                if bilingual: f.write(f"{translated}\n{src}\n\n")
                else: f.write(f"{translated}\n\n")

    def to_ass(self, output_path: str, bilingual: bool = False):
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write(ass_header + "\n")
            for seg in self.segments:
                start = self._srt_to_ass_time(seg.start_time)
                end = self._srt_to_ass_time(seg.end_time)
                translated = ASRData.clean_line(seg.translated_text)
                if not translated: continue
                text = f"{translated}\\N{seg.text}" if bilingual else translated
                text = text.replace('\n', '\\N')
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

# ==========================================
# 2. 基础翻译器基类 (改进：进度条 & 缓存清理)
# ==========================================

class BaseTranslator(ABC):
    def __init__(self, thread_num: int, batch_num: int, source_lang: str, target_lang: str, cache_dir: str):
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.source_lang = source_lang
        self.target_lang = target_lang
        # 设置缓存目录并清理过期缓存
        self.cache = Cache(cache_dir)
        # --- 改进2：自动清除已过期的缓存 ---
        self.cache.expire() 

        # 全局进度条变量：
        self.pbar = None
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
        
        # --- 改进1：增加 tqdm 进度条 ---
        self.pbar = tqdm(total=len(chunks), desc="[*] 翻译进度", unit="chunk")
        
        futures = {self.executor.submit(self._safe_translate_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(futures):
            try:
                result_chunk = future.result()
                for data in result_chunk:
                    translated_map[data.index] = data.translated_text
            except Exception as e:
                # 理论上 safe_translate 不会抛出异常，因为内部有 fallback
                print(f"\n[-] 批处理执行致命错误: {e}")
            finally:
                self.pbar.update(1)
        
        self.pbar.close()

        for seg in asr_data.segments:
            seg.translated_text = translated_map.get(seg.index, "")
        
        return asr_data

    def _safe_translate_chunk(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        key_data = [asdict(d) for d in chunk]
        cache_key = f"{self.__class__.__name__}:{generate_cache_key(key_data)}:{self.source_lang}:{self.target_lang}:{self.batch_num}"
        
        cached = self.cache.get(cache_key)
        if cached:
            self.pbar.write(f"[*] 命中缓存: {cache_key}")
            return [SubtitleProcessData(**d) for d in cached]

        result = self._translate_chunk(chunk)
        # 设置缓存，过期时间7天
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
# 3. LLM 翻译器实现 (改进：精准重试机制)
# ==========================================

class LLMTranslator(BaseTranslator):
    MAX_STEPS = 3  # Agent 纠错循环
    RETRY_LIMIT = 2 # 整个 Chunk 失败后的重试次数

    def __init__(self, api_config: dict, prompts: dict, thread_num: int, batch_num: int, 
                source_lang: str, target_lang: str, cache_dir: str, is_reflect: bool = False):
        
        super().__init__(thread_num, batch_num, source_lang, target_lang, cache_dir)
        # 设置LLM API接口
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
        """
        改进3：当 chunk 翻译失败后，识别出错条目，批量重重试，而不是直接逐条翻译。
        """
        # 1. 尝试整体翻译
        result_dict = self._attempt_chunk_translate(chunk)
        
        # 2. 检查结果，识别缺失的条目
        missing_indices = []
        for d in chunk:
            idx_str = str(d.index)
            if idx_str in result_dict:
                val = result_dict[idx_str]
                if self.is_reflect and isinstance(val, dict):
                    d.translated_text = val.get('native_translation', str(val))
                else:
                    d.translated_text = str(val)
            else:
                missing_indices.append(d)

        # 3. 如果有缺失条目，尝试针对这些条目进行一次“聚合重试”
        if missing_indices:
            self.pbar.write(f"\n[!] Chunk 中有 {len(missing_indices)} 条翻译失败，正在尝试精准重试...")
            retry_result = self._attempt_chunk_translate(missing_indices)
            
            # 再次填充
            still_missing = []
            for d in missing_indices:
                idx_str = str(d.index)
                if idx_str in retry_result:
                    val = retry_result[idx_str]
                    d.translated_text = val.get('native_translation', str(val)) if self.is_reflect and isinstance(val, dict) else str(val)
                else:
                    still_missing.append(d)
            
            # 4. 如果精准重试依然失败，最后才进入最后的兜底（逐条）
            if still_missing:
                self.pbar.write(f"\n[!] 重试后仍有 {len(still_missing)} 条失败，进入逐条翻译兜底模式。")
                self._translate_single_fallback(still_missing)
            
        return chunk

    def _attempt_chunk_translate(self, chunk: List[SubtitleProcessData]) -> Dict[str, Any]:
        """执行 Agent 循环翻译逻辑，返回成功解析的字典"""
        subtitle_dict = {str(d.index): d.original_text for d in chunk}
        prompt_type = "reflect" if self.is_reflect else "standard"
        system_prompt = self._get_prompt(prompt_type)
        
        try:
            return self._agent_loop(system_prompt, subtitle_dict)
        except Exception:
            return {}

    def _agent_loop(self, system_prompt: str, subtitle_dict: Dict[str, str]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(subtitle_dict, ensure_ascii=False)}
        ]
        
        last_resp_dict = {}
        for step in range(self.MAX_STEPS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content
                resp_dict = json_repair.loads(content)
                if not isinstance(resp_dict, dict):
                    raise ValueError("JSON is not a dict")
                
                last_resp_dict = resp_dict
                
                is_valid, error_msg = self._validate_response(resp_dict, subtitle_dict)
                if is_valid:
                    return resp_dict
                self.pbar.write(f"[!] 第{step + 1}次翻译中出现错误: {error_msg}")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Error: {error_msg}. Please fix and output the complete JSON object again."})
                time.sleep(1)
            except Exception as e:
                if step == self.MAX_STEPS - 1: raise e
                continue
        
        return last_resp_dict

    def _validate_response(self, resp_dict: Any, origin_dict: Dict[str, str]) -> Tuple[bool, str]:
        origin_keys = set(origin_dict.keys())
        resp_keys = set(resp_dict.keys())
        
        if not origin_keys.issubset(resp_keys):
            missing = origin_keys - resp_keys
            return False, f"Key 缺失: {missing}"
        
        if self.is_reflect:
            for k in origin_keys:
                v = resp_dict.get(k)
                if not isinstance(v, dict) or 'native_translation' not in v:
                    return False, f"Key '{k}' 格式不符合反思模式要求"
        
        return True, ""

    def _translate_single_fallback(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        system_prompt = self._get_prompt("single")
        self.pbar.write(f"[*] 进入逐条翻译模式...")
        for d in chunk:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": d.original_text}
                    ],
                    temperature=0.3,
                    timeout=15 # 单条翻译增加超时控制
                )
                d.translated_text = resp.choices[0].message.content.strip()
            except Exception:
                d.translated_text = d.original_text
        self.pbar.write(f"[*] 逐条翻译完成。")
        return chunk

# ==========================================
# 4. CLI 主程序 (保持原有逻辑，优化交互)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LLM Subtitle Translator--by liug")
    parser.add_argument("-i", "--input", required=True, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-s", "--source", default="Japanese", help="源语言")
    parser.add_argument("-l", "--lang", default="简体中文", help="目标语言")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--format", choices=['srt', 'ass'], default='srt', help="输出格式")
    parser.add_argument("--reflect", action="store_true", help="开启反思模式")
    parser.add_argument("--bilingual", action="store_true", help="输出双语对照")
    
    args = parser.parse_args()
    PROMPTS = {k: v for k, v in vars(prompts).items() if not k.startswith("__")}

    if not os.path.exists(args.config):
        config = {
            "api": {"api_key": "YOUR_API_KEY", "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
            "settings": {"thread_num": 2, "batch_num": 100, "cache_dir": "./.cache_sub_trans"}
        }
        with open(args.config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        print(f"[*] 已生成默认配置文件 {args.config}，请修改后运行。")
        return

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    try:
        asr_data = ASRData.from_srt(args.input)
    except Exception as e:
        print(f"[-] 读取失败: {e}")
        return

    translator = LLMTranslator(
        api_config=config['api'],
        prompts=PROMPTS,
        thread_num=config['settings'].get('thread_num', 4),
        batch_num=config['settings'].get('batch_num', 80),
        source_lang=args.source,
        target_lang=args.lang,
        cache_dir=config['settings'].get('cache_dir', './.cache_sub_trans'),
        is_reflect=args.reflect
    )

    print(f"[*] 任务开始: {args.source} -> {args.lang} | 模式: {'反思' if args.reflect else '标准'} | 并行数: {translator.thread_num} | 字幕数量/组: {translator.batch_num}")
    
    try:
        translated_data = translator.translate(asr_data)
        
        output_format = args.format
        if args.output:
            output_format = 'ass' if args.output.lower().endswith('.ass') else 'srt'
            output_file = args.output
        else:
            output_file = args.input.replace(".srt", f".{args.lang}.{output_format}")

        if output_format == 'ass':
            translated_data.to_ass(output_file, bilingual=args.bilingual)
        else:
            translated_data.to_srt(output_file, bilingual=args.bilingual)
            
        print(f"\n[+] 翻译成功！保存至: {output_file}")
    except Exception as e:
        print(f"\n[-] 翻译过程中出现异常: {e}")
    finally:
        translator.stop()

if __name__ == "__main__":
    main()
