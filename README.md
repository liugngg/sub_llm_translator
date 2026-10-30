# LLM 字幕翻译工具

通过 OpenAI 兼容的 Chat Completions 或 Responses 接口翻译 SRT 字幕，也支持导出 ASS、双语字幕、批量处理和本地缓存。

## 环境与安装

需要 Python 3.12 或更高版本。在项目目录执行：

```powershell
uv sync
```

首次运行若找不到 `config.yaml`，程序会在程序目录生成模板并退出。源码运行时程序目录是 `sub_trans.py` 所在目录；打包运行时是 exe 所在目录。显式指定 `-c` 时使用该路径。

## 配置文件

```yaml
apis:
  openai:
    api_key: "YOUR_API_KEY"
    base_url: "https://api.openai.com" # 也可写成 https://api.openai.com/v1
    model: "gpt-4o"
    format: chat_completions # 可选 responses
  compatible:
    api_key: "YOUR_API_KEY"
    base_url: "https://provider.example.com/v1"
    model: "your-model-name"
default_api: openai
settings:
  thread_num: 2
  batch_num: 10
  cache_dir: "./.cache_sub_trans"
  temperature: 0.3
  timeout: 45
  request_interval: 1
  merge_interval: 0.7
  max_duration: 8
replacements: {}
```

`base_url` 支持写服务根地址（程序自动补 `/v1`）或已经带 `/v1` 的地址。`format` 默认为 `chat_completions`，也可设为 `responses`。前者请求 `/chat/completions`，后者使用 SDK 的 `/responses` 接口。请确认服务提供商支持所选接口及 JSON 输出格式。例如 `https://host.example/api/v1` 会请求 `https://host.example/api/v1/chat/completions`。程序不会检测服务是否在线，也无法修复网络、DNS、代理或证书问题。

`apis` 可配置多个服务。每个服务都需要 `api_key`、`base_url` 和 `model`。`default_api` 指定默认服务；也可通过 `-a/--api` 选择其他服务。密钥请只保存在本机配置或环境变量中，不要提交到版本库。

配置多个服务时，程序会自动轮询：先使用 `-a` 指定的服务（未指定时用 `default_api`），其余服务按配置文件顺序作为后备。某个批次遇到认证错误、模型不存在或整批内容审查拒绝时，会切换到下一个服务重试；某个服务翻译成功后，后续请求优先使用它。密钥或地址缺失的服务会被跳过并给出警告。命中缓存时不会发起请求，因此也不会触发切换。

| 配置项 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `settings.thread_num` | `2` | 并发翻译请求数。遇到服务限流时调低。 |
| `settings.batch_num` | `10` | 每个请求最多包含的字幕条目数。调大可减少请求次数，但会增加单次请求上下文和失败重试成本。 |
| `settings.cache_dir` | `./.cache_sub_trans` | 翻译缓存目录。成功批次缓存 7 天；包含审查标记的结果缓存 1 小时；网络等失败不缓存。缓存按服务地址、模型、接口、温度和提示词隔离。 |
| `settings.temperature` | `0.3` | 模型采样温度；具体支持范围取决于服务。 |
| `settings.timeout` | `45` | 单次 API 请求超时秒数，必须为正整数。 |
| `settings.request_interval` | `1` | 相邻两次 API 请求的最小间隔（秒），必须为非负数，`0` 表示不限速。多线程共享同一条时间线，用于避免高并发触发限流。 |
| `settings.merge_interval` | `0.7` | 合并相邻字幕的最大时间间隔（秒），必须为非负数。 |
| `settings.max_duration` | `8` | 合并后字幕段允许的最大时长（秒），必须为正数。 |
| `replacements` | `{}` | 翻译前的文本替换映射，键为原文，值为替换文本；支持项目已有的正则替换规则。启动时会检查正则表达式语法，无效的将被忽略并发出警告。 |

**配置验证：** 程序启动时会验证配置的合法性。`temperature` 必须在 0-2 之间，`thread_num`、`batch_num`、`timeout` 必须为正整数，`merge_interval`、`max_duration`、`request_interval` 必须为有效数值。

## 命令行参数

基本格式：

```powershell
python sub_trans.py -i <输入文件或目录> [选项]
```

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `-i`, `--input` | 必填 | 输入 SRT 文件，或递归搜索 SRT 文件的目录。 |
| `-o`, `--output` | 自动生成 | 输出文件路径或目录。单文件输入可给文件名；多文件输入时必须给目录。未指定时，SRT 输出为 `<原文件名>.<目标语言>.srt`，ASS 输出为 `<原文件名>.ass`；指定输出目录时也遵循此命名规则。 |
| `-s`, `--source` | `Japanese` | 字幕源语言名称，会传入提示词。 |
| `-l`, `--lang` | `简体中文` | 目标语言名称，也用于默认输出文件名。 |
| `-c`, `--config` | 程序目录下的 `config.yaml` | YAML 配置文件路径。源码运行时默认为脚本目录，打包运行时默认为 exe 目录。 |
| `-a`, `--api` | 配置中的 `default_api` | 选择 `apis` 下的服务配置名称。 |
| `--format` | `srt` | 输出格式，可选 `srt` 或 `ass`。显式指定 `--output` 为 `.ass` 文件时也会按扩展名输出 ASS。 |
| `--bilingual` | 关闭 | 输出原文与译文双语字幕。 |
| `-h`, `--help` | — | 显示参数帮助并退出。 |

### 请求失败与审查处理

- 网络、限流或格式错误最多尝试 3 次，两次重试前分别等待 1 秒、2 秒。SDK 自身也可能重试网络请求，因此 HTTP 请求数可能更多。
- 配置了多个服务时，认证错误、模型不存在或整批审查拒绝会切换到下一个服务重新请求；所有服务均拒绝时才对半拆分批次隔离处理，单条仍被拒绝则输出 `【云端审查】原文`。JSON 内的明确拒绝说明只标记对应条目，保留其他译文。
- 普通失败耗尽重试后输出 `【翻译失败】原文`，其他批次继续处理。所有服务均认证失败时终止当前文件；目录模式会继续尝试后续文件。
- 内容审查处理不会保证云端服务接受内容；服务商的政策和模型行为仍然适用。
- 以上默认值用于新生成配置及缺省字段，已有配置中明确设置的值保持有效。

### 漏翻检测与标记

日译中时，程序会自动检测译文中疑似未翻译的内容：

- **检测规则：** 译文中保留较多日文假名（平假名或片假名），或去除标点后与原文完全相同
- **处理流程：** 检出疑似条目后单独重译一次；重译后仍未通过检查或请求失败时，该字幕译文会以 `【原文】` 格式标记输出
- **优化措施：** 短字幕（<10 字符）使用更宽松的阈值，避免人名、口语被误判

示例：

```powershell
# 翻译单个文件，使用默认配置和服务
python sub_trans.py -i input.srt

# 指定配置、API 服务和语言，并保存到指定文件
python sub_trans.py -i input.srt -c .\config.yaml -a compatible -s Japanese -l 简体中文 -o output.srt

# 递归处理目录，输出 ASS 双语字幕
python sub_trans.py -i .\subtitles -o .\translated --format ass --bilingual
```

## 打包

打包工具只属于构建环境，不再作为程序运行依赖安装。先安装构建依赖：

```powershell
uv sync --group build
```

使用 PyInstaller 的目录模式（`--onedir`）打包成一个可分发目录：

```powershell
uv run --with pyinstaller pyinstaller --clean --noconfirm --onedir --console --name sub_llm_trans sub_trans.py
```

构建完成后，可执行文件位于 `dist\sub_llm_trans\sub_llm_trans.exe`。将整个 `dist\sub_llm_trans` 目录一起分发；首次运行前把 `config.yaml` 放在该目录中，缺少配置时程序会在 exe 所在目录生成模板。

如需单文件：

```powershell
uv run --with pyinstaller pyinstaller --clean --noconfirm --onefile --console --name sub_llm_trans sub_trans.py
```

PyInstaller 需要包含 Python 运行时以及 `openai`、HTTP 客户端、YAML、缓存等运行依赖，因此产物仍会明显大于源代码。避免在打包环境中安装不需要的开发依赖，并优先使用目录模式；UPX 可选压缩，但需要单独安装，且某些安全软件可能误报。构建后首次运行前，将 `config.yaml` 放在可执行文件所在目录；若文件不存在，程序会在那里生成模板。

## 开发检查

```powershell
uv run python -m compileall sub_trans.py prompts.py
uv run python -m unittest discover -s test -p "test_*.py" -v
# 单个测试（从项目根目录运行，避开系统 Python 的同名 test 包）
uv run python -m unittest discover -s test -p test_translation.py -k test_ordinary_dialogue_is_not_a_refusal -v
```

测试涵盖字幕处理及使用模拟 API 的翻译工作流，不调用真实服务。涉及翻译逻辑的修改后，还需使用获准提交云端的小型 SRT 手动验证真实服务兼容性、缓存命中以及所需输出格式。
