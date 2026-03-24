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

class ASRDataSeg:
    def __init__(self, text: str, start_time: str, end_time: str, index: int, translated_text: str = ""):
        self.text = text
        self.start_time = start_time # 格式: 00:00:00,000
        self.end_time = end_time
        self.index = index
        self.translated_text = translated_text

class ASRData:
    def __init__(self, segments: List[ASRDataSeg]):
        self.segments = sorted(segments, key=lambda x: x.index)

    @staticmethod    # 清理字幕文件
    def clean_line(text: str) -> str:
        """
        1. 去掉首尾【一般】标点
        2. 同时清除收尾标点，结尾的问号需要保留
        """
        text = text.strip()
        if not text:
            return ''

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
            # text = match.group(3).strip()
            text = ASRData.clean_line(match.group(3))
            segments.append(ASRDataSeg(text, times[0], times[1], idx))
        
        if not segments:
            raise ValueError("无法解析 SRT 文件，请检查格式是否标准。")
        return ASRData(segments)

    def _srt_to_ass_time(self, srt_time: str) -> str:
        """将 SRT 时间格式 00:00:00,000 转换为 ASS 格式 0:00:00.00"""
        # 替换逗号为点
        t = srt_time.replace(',', '.')
        # 去掉小时前面的第一个0 (如果是 00:...)
        if t.startswith('0'):
            t = t[1:]
        # 毫秒三位转两位 (123 -> 12)
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
                translated =  ASRData.clean_line(seg.translated_text)
                if bilingual:
                    f.write(f"{translated}\n{src}\n\n")
                else:
                    f.write(f"{translated if translated else src}\n\n")

    def to_ass(self, output_path: str, bilingual: bool = False):
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            f.write(ass_header + "\n")
            for seg in self.segments:
                start = self._srt_to_ass_time(seg.start_time)
                end = self._srt_to_ass_time(seg.end_time)
                
                # 文本处理
                src = ASRData.clean_line(seg.text)
                translated =  ASRData.clean_line(seg.translated_text)
                if bilingual:
                    # ASS 使用 \N 表示换行
                    text = f"{translated}\\N{src}"
                else:
                    text = translated if translated else src
                
                # 清洗文本中的换行符为ASS换行符
                text = text.replace('\n', '\\N')
                
                f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

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
# 5. CLI 主程序
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="LLM Subtitle Translator--by liug")
    parser.add_argument("-i", "--input", required=True, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径 (支持 .srt 或 .ass)")
    parser.add_argument("-s", "--source", default="Japanese", help="源语言 (默认: Japanese)")
    parser.add_argument("-l", "--lang", default="简体中文", help="目标语言 (默认: 简体中文)")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--format", choices=['srt', 'ass'], default='srt', help="输出格式 (默认: srt)")
    parser.add_argument("--reflect", action="store_true", help="开启反思模式")
    parser.add_argument("--bilingual", action="store_true", help="输出双语对照")
    
    args = parser.parse_args()
    # 获取提示词并组成词典,过滤掉系统内置属性（以 __ 开头的）
    PROMPTS = {k: v for k, v in vars(prompts).items() if not k.startswith("__")}

    # 加载或创建配置文件 (注意：这里不再在 yaml 中存储 prompts)
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
            }
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
        prompts=PROMPTS, # 使用 prompts.py 中的内容
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
        
        # 确定输出路径和格式
        output_format = args.format
        if args.output:
            if args.output.lower().endswith('.ass'):
                output_format = 'ass'
            elif args.output.lower().endswith('.srt'):
                output_format = 'srt'
            output_file = args.output
        else:
            output_file = args.input.replace(".srt", f".{args.lang}.{output_format}")

        # 保存结果
        if output_format == 'ass':
            translated_data.to_ass(output_file, bilingual=args.bilingual)
        else:
            translated_data.to_srt(output_file, bilingual=args.bilingual)
            
        print(f"[+] 翻译成功！保存至: {output_file}")
    except Exception as e:
        print(f"[-] 翻译过程中出现异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        translator.stop()

if __name__ == "__main__":
    main()
