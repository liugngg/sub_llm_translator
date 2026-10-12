# LLM 字幕翻译工具

通过 OpenAI 兼容的聊天补全接口翻译 SRT 字幕，也支持导出 ASS、双语字幕、批量处理和本地缓存。

## 环境与安装

需要 Python 3.12 或更高版本。在项目目录执行：

```powershell
uv sync
```

首次运行若找不到 `config.yaml`，程序会在当前目录生成模板并退出。填写 API 信息后再运行。

## 配置文件

```yaml
apis:
  openai:
    api_key: "YOUR_API_KEY"
    base_url: "https://api.openai.com" # 也可写成 https://api.openai.com/v1
    model: "gpt-4o"
  compatible:
    api_key: "YOUR_API_KEY"
    base_url: "https://provider.example.com/v1"
    model: "your-model-name"
default_api: openai
settings:
  thread_num: 4
  batch_num: 40
  cache_dir: "./.cache_sub_trans"
  temperature: 0.7
  timeout: 30
  merge_interval: 0.7
  max_duration: 8
replacements: {}
```

`base_url` 支持写服务根地址（程序自动补 `/v1`）或已经带 `/v1` 的地址。实际请求会在其后追加 `/chat/completions`，所以它必须是 OpenAI 兼容的 Chat Completions API 地址。例如 `https://host.example/api/v1` 会请求 `https://host.example/api/v1/chat/completions`。程序不会检测服务是否在线，也无法修复网络、DNS、代理或证书问题。

`apis` 可配置多个服务。每个服务都需要 `api_key`、`base_url` 和 `model`。`default_api` 指定默认服务；也可通过 `-a/--api` 选择其他服务。密钥请只保存在本机配置或环境变量中，不要提交到版本库。

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `settings.thread_num` | `4` | 并发翻译请求数。遇到服务限流时调低。 |
| `settings.batch_num` | `40` | 每个请求最多包含的字幕条目数。调大可减少请求次数，但会增加单次请求上下文和失败重试成本。 |
| `settings.cache_dir` | `./.cache_sub_trans` | 翻译缓存目录。成功的批次缓存 7 天。重复运行相同内容可复用缓存。 |
| `settings.temperature` | `0.7` | 模型采样温度；具体支持范围取决于服务。 |
| `settings.timeout` | `30` | 单次 API 请求超时秒数，必须为正整数。 |
| `settings.merge_interval` | `0.7` | 合并相邻字幕的最大时间间隔（秒）。 |
| `settings.max_duration` | `8` | 合并后字幕段允许的最大时长（秒）。 |
| `replacements` | `{}` | 翻译前的文本替换映射，键为原文，值为替换文本；支持项目已有的正则替换规则。 |

## 命令行参数

基本格式：

```powershell
python sub_trans.py -i <输入文件或目录> [选项]
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-i`, `--input` | 必填 | 输入 SRT 文件，或递归搜索 SRT 文件的目录。 |
| `-o`, `--output` | 自动生成 | 输出文件路径或目录。单文件输入可给文件名；多文件输入时必须给目录。未指定时，SRT 输出为 `<原文件名>.<目标语言>.srt`，ASS 输出为 `<原文件名>.ass`。 |
| `-s`, `--source` | `Japanese` | 字幕源语言名称，会传入提示词。 |
| `-l`, `--lang` | `简体中文` | 目标语言名称，也用于默认输出文件名。 |
| `-c`, `--config` | `config.yaml` | YAML 配置文件路径。 |
| `-a`, `--api` | 配置中的 `default_api` | 选择 `apis` 下的服务配置名称。 |
| `--format` | `srt` | 输出格式，可选 `srt` 或 `ass`。显式指定 `--output` 为 `.ass` 文件时也会按扩展名输出 ASS。 |
| `--reflect` | 关闭 | 使用反思提示词，通常能改善语境和口语表达，但增加 token 消耗。 |
| `--bilingual` | 关闭 | 输出原文与译文双语字幕。 |
| `-h`, `--help` | — | 显示参数帮助并退出。 |

示例：

```powershell
# 翻译单个文件，使用默认配置和服务
python sub_trans.py -i input.srt

# 指定配置、API 服务和语言，并保存到指定文件
python sub_trans.py -i input.srt -c .\config.yaml -a compatible -s Japanese -l 简体中文 -o output.srt

# 递归处理目录，输出 ASS 双语字幕并开启反思模式
python sub_trans.py -i .\subtitles -o .\translated --format ass --bilingual --reflect
```

## 打包

打包工具只属于构建环境，不再作为程序运行依赖安装。先安装构建依赖：

```powershell
uv sync --group build
```

推荐使用 PyInstaller 的目录模式。通常比单文件启动更快，也避免每次运行解压；发布时将整个目录一起分发：

```powershell
uv run pyinstaller --clean --noconfirm --onedir --console --name sub_llm_trans sub_trans.py
```

如需单文件：

```powershell
uv run pyinstaller --clean --noconfirm --onefile --console --name sub_llm_trans sub_trans.py
```

PyInstaller 需要包含 Python 运行时以及 `openai`、HTTP 客户端、YAML、缓存等运行依赖，因此产物仍会明显大于源代码。避免在打包环境中安装不需要的开发依赖，并优先使用目录模式；UPX 可选压缩，但需要单独安装，且某些安全软件可能误报。构建后首次运行前，将 `config.yaml` 放在可执行文件的工作目录中。

## 开发检查

```powershell
python -m compileall sub_trans.py prompts.py
```

项目目前没有自动化测试套件。涉及翻译逻辑的修改后，请用小型 SRT 文件手动验证普通行、空行、缓存命中以及所需输出格式。
