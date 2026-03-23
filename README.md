# LLM字幕翻译工具--liug

## 1. 主要功能：

1. 一个通过LLM进行字幕翻译的CLI程序
2. 输入为待翻译的字幕文件，可以指定字幕源语言(默认日语)、目标语音(默认为中文)。输出为翻译后的字幕文件。
3. 提供配置文件可以设置LLM API接口、API-key以及模型名称、Prompt 模板等内容。

## 2. 具体特点：

- 多线程并行：程序会根据 config.yaml 中的 thread_num 开启多个并发请求。如果你的 API 账户有频率限制（Rate Limit），请将此值调小（例如 2 或 1）。
- 缓存机制：
  - 该 CLI 程序使用了你提供的 Cache 模块。
  - 翻译结果会存储在本地（由 CACHE_PATH 决定，默认通常在用户临时目录）。
  - 如果任务中途失败，再次运行相同的任务时，已翻译的块将直接从硬盘读取，不消耗 Token。
- Agent Loop 纠错：
  - LLMTranslator 会自动检查 LLM 返回的 JSON 是否包含所有请求的 index。
  - 如果 LLM “偷懒”漏掉了几行，代码会自动向 LLM 发送错误反馈并要求重试。
- 反思模式 (Reflect)：
  - 开启后，LLM 会先翻译一遍，然后检查语境，最后输出一个地道的版本。
  - 这种模式消耗两倍以上的 Token，但对于电影俚语或复杂句子的翻译质量极高。
- 单条兜底：
  - 如果某一组字幕块因为内容敏感或格式问题始终无法通过 JSON 校验，程序会降级为 _translate_chunk_single，即逐条进行翻译，确保任务不会中断。

## 3. 打包命令

- 生成单文件格式
  `pyinstaller -i liug.ico -F -w main.py --clean -n LLM字幕翻译`

- 生成单文件格式（Nuitka --onefile自动压缩）
- 如果你还没有在当前环境中安装 nuitka，你可以使用 --with 参数让 uv 临时安装并运行它，而无需手动 pip install
  `uv run --with nuitka python -m nuitka --mingw64 --onefile --lto=yes --show-progress --output-dir=dist --remove-output --windows-console-mode=force main.py`

- 生成单文件格式（Nuitka --使用upx 压缩）
  `uv run --with nuitka python -m nuitka --mingw64 --onefile --onefile-no-compression --plugin-enable=upx --lto=yes --show-progress --output-dir=dist --remove-output --windows-console-mode=force main.py`

## 4. 作者

- [liugngg (GitHub地址)](https://github.com/liugngg)
