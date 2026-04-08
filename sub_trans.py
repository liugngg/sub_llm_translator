import os
import json
import yaml
import hashlib
import argparse
import re
import time
import atexit
import sys # 导入 sys 模块用于程序退出
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

import openai
from openai import OpenAI, RateLimitError, AuthenticationError, NotFoundError, APITimeoutError # 导入更具体的异常
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
Style: Default,Microsoft YaHei,60,&H0000FFFF,&H000000FF,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

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
BLANK_RE = re.compile(r'^[ ,.，。！、!?？：；;—\-\–…\"''~「」『』啊嗬嗯哈唔哎呼咿呜呀西咻昂呐恩库莫伊阿咕哒喽呗嘛哟哇呃哦啦唉欸诶喔哼嘿喂干燥咚哔んっはいふぁっあうちゅちあたえ]*$', flags=re.UNICODE|re.MULTILINE)
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
# 2. 基础翻译器基类 (改进：进度条 & 缓存清理 & 新的错误处理/重试机制)
# ==========================================

class TranslationError(Exception):
    """自定义异常，用于表示翻译过程中LLM返回结果不符合预期"""
    pass

class BaseTranslator(ABC):
    MAX_CHUNK_RETRIES = 2 # 针对单个chunk的最大重试次数

    def __init__(self, thread_num: int, batch_num: int, source_lang: str, target_lang: str, cache_dir: str, timeout: int):
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.timeout = timeout # 新增 timeout 参数
        # 设置缓存目录并清理过期缓存
        self.cache = Cache(cache_dir)
        self.cache.expire() 

        # 全局进度条变量：
        self.pbar = None
        self.executor = ThreadPoolExecutor(max_workers=thread_num)
        self.is_running = True
        atexit.register(self.stop)

    def write(self, msg: str):
        if self.pbar: 
            self.pbar.write(msg)
        else:
            print(msg)
    
    def translate(self, asr_data: ASRData) -> ASRData:
        translate_data_list = [
            SubtitleProcessData(index=seg.index, original_text=seg.text)
            for seg in asr_data.segments
        ]
        
        chunks = [translate_data_list[i:i + self.batch_num] for i in range(0, len(translate_data_list), self.batch_num)]
        translated_map = {}
        
        self.pbar = tqdm(total=len(chunks), desc="[*] 翻译进度", unit="chunk")
        
        futures = {self.executor.submit(self._process_chunk_with_retry, chunk): chunk for chunk in chunks}
        
        for future in as_completed(futures):
            try:
                result_chunk = future.result()
                for data in result_chunk:
                    translated_map[data.index] = data.translated_text
            except Exception as e:
                # _process_chunk_with_retry 内部已经处理了重试和退出逻辑，
                # 如果到这里抛出异常，说明是其他未知且致命的线程池错误。
                self.write(f"\n[-] 批处理执行遇到未知致命错误: {e}")
                sys.exit(1) # 发现严重错误则退出
            finally:
                self.pbar.update(1)
        
        self.pbar.close()

        for seg in asr_data.segments:
            seg.translated_text = translated_map.get(seg.index, "")
        
        return asr_data

    def _process_chunk_with_retry(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        """
        处理单个 chunk，包含缓存、错误处理和重试逻辑。
        """
        cache_key = self._generate_chunk_cache_key(chunk)
        
        cached = self.cache.get(cache_key)
        if cached:
            self.write(f"[*] 命中缓存: {cache_key}")
            return [SubtitleProcessData(**d) for d in cached]

        # 重试逻辑
        for attempt in range(1, self.MAX_CHUNK_RETRIES + 1):
            try:
                result = self._translate_chunk(chunk)
                # 如果成功，设置缓存并返回
                self.cache.set(cache_key, [asdict(d) for d in result], expire=86400 * 7)
                return result
            
            except (APITimeoutError, TranslationError) as e:
                if attempt < self.MAX_CHUNK_RETRIES:
                    self.write(f"\n[-] 翻译 chunk 失败 (尝试 {attempt}/{self.MAX_CHUNK_RETRIES})，错误: {e}。1秒后重试...")
                    time.sleep(1) # 等待1秒后重试
                else:
                    self.write(f"\n[-] 翻译 chunk 失败 (已达最大重试次数 {self.MAX_CHUNK_RETRIES})，错误: {e}。程序将退出。")
                    self.stop()
                    time.sleep(2)
                    sys.exit(1) # 达到最大重试次数后退出
            except Exception as e:
                self.write(f"\n[-] 翻译 chunk 遇到未知错误: {e}。程序将退出。")
                self.stop()
                time.sleep(2)
                sys.exit(1) # 其他意外错误，也直接退出

        # 理论上不会执行到这里，因为达到最大重试次数会调用 sys.exit()
        return [] 

    @abstractmethod
    def _translate_chunk(self, chunk: List[SubtitleProcessData]) -> List[SubtitleProcessData]:
        """抽象方法，子类实现具体的翻译逻辑，如果失败应抛出 TranslationError 或 OpenAI API 异常"""
        pass

    def _generate_chunk_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        key_data = [asdict(d) for d in chunk]
        return f"{self.__class__.__name__}:{generate_cache_key(key_data)}:{self.source_lang}:{self.target_lang}:{self.batch_num}"

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

    def __init__(self, api_config: dict, prompts: dict, thread_num: int, batch_num: int, 
                source_lang: str, target_lang: str, cache_dir: str, is_reflect: bool = False, 
                temperature: float = 0.6, timeout: int = 60): # 增加 timeout 参数及其默认值
        
        super().__init__(thread_num, batch_num, source_lang, target_lang, cache_dir, timeout)
        # 设置LLM API接口
        self.client = OpenAI(api_key=api_config['api_key'], base_url=api_config['base_url'])
        self.model = api_config['model']
        self.prompts = prompts
        self.is_reflect = is_reflect
        self.temperature = temperature

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
                    # 在反思模式下，尝试从 'native_translation' 获取，否则使用原始值
                    d.translated_text = val.get('native_translation', str(val))
                else:
                    d.translated_text = str(val)
            return chunk
        except (RateLimitError, APITimeoutError, TranslationError) as e:
            # 直接将这些错误向上抛出，由 _process_chunk_with_retry 处理
            raise e
        except Exception as e:
            # 捕获其他所有未预料的错误，并包装成 TranslationError
            raise TranslationError(f"在LLM翻译或解析结果时发生未知错误: {e}") from e

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
                    temperature=self.temperature, # 使用实例的 temperature
                    response_format={"type": "json_object"},
                    timeout=self.timeout # 使用配置中的 TIMEOUT
                )
            except (AuthenticationError, NotFoundError) as e:
                # 这些致命错误直接退出
                self.write(f"\n[-] 发生致命错误：OpenAI API 认证/资源错误，无法继续: {e}")
                self.stop()
                self.write(f"\n[-] 程序正在退出。。。")
                time.sleep(2)
                sys.exit(1) # 致命错误，直接退出
            except (RateLimitError, APITimeoutError) as e:
                # 这些错误向上抛出
                raise e
            except Exception as e:
                # 其他任何 OpenAI 库可能抛出的错误，视为需要重试的 TranslationError
                raise TranslationError(f"OpenAI API 调用发生错误: {e}") from e

            content = response.choices[0].message.content
            
            try:
                # 使用 json_repair 处理可能不完整的 JSON
                resp_dict = json_repair.loads(content)
                last_resp_dict = resp_dict
            except json.JSONDecodeError as e:
                # 如果 JSON 格式不正确，视为需要重试的 TranslationError
                error_msg = f"LLM返回的JSON格式不正确: {e}. 原始内容: {content[:200]}..."
                self.write(f"\n[-] {error_msg}")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Error: {error_msg}. Please fix and output the complete JSON object again."})
                continue # 进入下一次 agent 循环尝试纠正

            is_valid, error_msg = self._validate_response(resp_dict, subtitle_dict)
            if is_valid:
                return resp_dict
            
            # 如果验证失败，向模型发送纠错信息
            self.write(f"\n[-] LLM返回结果验证失败 (尝试 {step+1}/{self.MAX_STEPS}): {error_msg}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Error: {error_msg}. Please fix and output the complete JSON object again."})
        
        # 达到最大纠错次数仍未成功，抛出 TranslationError
        raise TranslationError(f"LLM在 {self.MAX_STEPS} 次纠错后仍未能返回正确格式的结果。最后一个结果: {last_resp_dict}")

    def _validate_response(self, resp_dict: Any, origin_dict: Dict[str, str]) -> Tuple[bool, str]:
        if not isinstance(resp_dict, dict):
            return False, f"返回结果不是一个字典类型，而是 {type(resp_dict)}"

        origin_keys = set(origin_dict.keys())
        resp_keys = set(resp_dict.keys())
        
        # 确保所有原始键都在返回结果中
        if not origin_keys.issubset(resp_keys):
            missing = origin_keys - resp_keys
            return False, f"返回结果缺少以下键: {missing}"
        
        # 确保返回的字典没有多余的键 (可选，但有助于更严格的结构检查)
        if not resp_keys.issubset(origin_keys):
            extra = resp_keys - origin_keys
            # 这里可以根据需要决定是返回 False 还是仅发出警告。
            # 为了更严格，我们视为错误。
            return False, f"返回结果包含不应存在的额外键: {extra}"

        if self.is_reflect:
            for k in origin_keys:
                v = resp_dict.get(k)
                if not isinstance(v, dict) or 'native_translation' not in v or not isinstance(v['native_translation'], str):
                    return False, f"键 '{k}' 的值不符合反思模式的要求 (需为包含 'native_translation' 键的字典，且 'native_translation' 的值需为字符串)"
        else:
            for k in origin_keys:
                v = resp_dict.get(k)
                if not isinstance(v, str):
                    return False, f"键 '{k}' 的值不符合标准模式的要求 (需为字符串类型)"
        
        return True, ""

# ==========================================
# 4. CLI 主程序 (保持原有逻辑，优化交互)
# ==========================================

def main():
    def find_srt_files(input_path: str) -> List[str]:
        srt_files = []
        if not os.path.exists(input_path):
            # raise FileNotFoundError(f"输入路径 {input_path} 不存在")
            return []
        if os.path.isfile(input_path):
            if input_path.lower().endswith(".srt") and "简体中文" not in input_path:
                srt_files.append(input_path)
        elif os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.endswith(".srt") and "简体中文" not in file:
                        full_path = os.path.join(root, file)
                        srt_files.append(os.path.abspath(full_path))
        return srt_files

    parser = argparse.ArgumentParser(description="LLM Subtitle Translator--by liug")
    parser.add_argument("-i", "--input", required=True, help="输入 SRT 文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径")
    parser.add_argument("-s", "--source", default="Japanese", help="源语言")
    parser.add_argument("-l", "--lang", default="简体中文", help="目标语言")
    parser.add_argument("-c", "--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("-a", "--api", help="指定使用的 API 名称 (对应 config.yaml 中的 apis 键名)") # 新增参数
    parser.add_argument("--format", choices=['srt', 'ass'], default='srt', help="输出格式")
    parser.add_argument("--reflect", action="store_true", help="开启反思模式")
    parser.add_argument("--bilingual", action="store_true", help="输出双语对照")
    
    args = parser.parse_args()
    # 获取翻译的提示词
    PROMPTS = {k: v for k, v in vars(prompts).items() if not k.startswith("__")}

    # 1. 获取 srt 文件列表
    srt_files = find_srt_files(args.input)
    if not srt_files:
        print(f"[-] 输入路径 {args.input} 不包含任何 SRT 文件")
        sys.exit(1)
    else:
        print(f"[*] 找到 {len(srt_files)} 个 SRT 文件")


    # 2. 检查并加载配置文件
    if not os.path.exists(args.config):
        # 默认生成新结构的配置
        default_config = {
            "apis": {
                "openai": {"api_key": "YOUR_API_KEY", "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
                "gemini": {"api_key": "YOUR_API_KEY", "base_url": "https://kaka.liugngg.top/v1", "model": "gemini-1.5-flash"}
            },
            "default_api": "openai",
            "settings": {
                "thread_num": 4, 
                "batch_num": 40, 
                "cache_dir": "./.cache_sub_trans", 
                "temperature": 0.7,
                "timeout": 30 
            }
        }
        with open(args.config, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True)
        print(f"[*] 已生成默认配置文件 {args.config}，请修改后运行。")
        sys.exit(0)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 3. 确定使用哪一个 API 配置
    available_apis = config.get('apis', {})
    if not available_apis:
        print("[-] 错误：配置文件中没有找到 'apis' 配置项。")
        sys.exit(1)
    # 优先级：命令行参数 > 配置文件中的 default_api > 第一个可用的 API
    api_name = args.api or config.get('default_api')
    if not api_name or api_name not in available_apis:
        if args.api:
            print(f"[-] 错误：指定的 API '{args.api}' 不在配置文件中。")
            print(f"[*] 可用的 API 有: {', '.join(available_apis.keys())}")
            sys.exit(1)
        else:
            api_name = list(available_apis.keys())[0]
            print(f"[*] 未指定 API 且无默认设置，自动选择第一个: {api_name}")
    selected_api_config = available_apis[api_name]
    print(f"[*] 已选择 API 配置: [{api_name}] (Model: {selected_api_config.get('model')})")
    

    # 4. 初始化翻译器 (传入 selected_api_config)
    translator = LLMTranslator(
        api_config=selected_api_config,
        prompts=PROMPTS,
        thread_num=config['settings'].get('thread_num', 4),
        batch_num=config['settings'].get('batch_num', 40),
        source_lang=args.source,
        target_lang=args.lang,
        cache_dir=config['settings'].get('cache_dir', './.cache_sub_trans'),
        is_reflect=args.reflect,
        temperature=config['settings'].get('temperature', 0.7),
        timeout=config['settings'].get('timeout', 30)
    )

    # 开始翻译：
    print(f"[*] 任务开始: {args.source} -> {args.lang} | 模式: {'反思' if args.reflect else '标准'} | 并行数: {translator.thread_num}")
    print("="*80)
    for i, srt_file in enumerate(srt_files, 1):
        # ... (循环内部逻辑保持不变)
        translator.write(f"[*] 正在处理文件 {i}/{len(srt_files)}: {srt_file}")
        try:
            asr_data = ASRData.from_srt(srt_file)
            translated_data = translator.translate(asr_data)
            
            # ... (保存逻辑)
            output_format = args.format
            if args.output:
                if os.path.isdir(args.output):
                    input_filename = os.path.basename(srt_file)
                    name_without_ext = os.path.splitext(input_filename)[0]
                    output_file = os.path.join(args.output, f"{name_without_ext}.{args.lang}.{output_format}")
                else:
                    output_file = args.output
                    output_format = 'ass' if args.output.lower().endswith('.ass') else 'srt'
            else:
                output_file = srt_file.replace(".srt", f".{args.lang}.{output_format}")
            if output_format == 'ass':
                translated_data.to_ass(output_file, bilingual=args.bilingual)
            else:
                translated_data.to_srt(output_file, bilingual=args.bilingual)
            translator.write(f"\n[+] 翻译成功！保存至: {output_file}")
        except Exception as e:
            translator.write(f"\n[-] 翻译过程中出现未预期异常: {e}")
            continue
    translator.stop()
    print("="*80)
    print("\n[*] 所有任务已完成！")

if __name__ == "__main__":
    main()
