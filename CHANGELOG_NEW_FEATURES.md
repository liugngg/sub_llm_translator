# 新功能说明

## 功能 1: 快速模式批次重译机制

### 背景
在快速模式下，程序会跳过疑似未翻译的字幕条目以提高效率。但如果单个批次中漏翻的条目过多，可能表示整批翻译质量不佳，需要重新翻译。

### 实现
- 新增配置项 `settings.fast_mode_retry_threshold`（默认值：3）
- 在快速模式下，如果单批次跳过的字幕条数超过该阈值，程序会：
  1. 清除该批次的缓存
  2. 重新翻译整批字幕（仅重试一次）
  3. 重新检查翻译结果，仍然漏翻的条目将被跳过

### 配置示例
```yaml
settings:
  fast_mode: true
  fast_mode_retry_threshold: 3  # 单批次漏翻超过 3 条时重译整批
```

### 日志输出
```
[!] 快速模式：批次漏翻 5 条，超过阈值 3，重译整批
[!] 快速模式：重译后仍跳过疑似漏翻字幕 42
```

---

## 功能 2: 翻译前 API 选择交互

### 背景
用户可能配置了多个 LLM API 服务，希望在翻译开始前选择使用哪个模型，而不是每次都通过命令行参数指定。

### 实现
- 新增配置项 `settings.api_selection_timeout`（默认值：3 秒）
- 程序启动时，如果未通过 `-a` 参数指定 API：
  1. 列出所有配置的 API 及其模型、URL
  2. 等待用户输入选择（支持序号或名称）
  3. 超时后自动选择 `default_api` 配置的默认服务

### 配置示例
```yaml
apis:
  openai:
    api_key: "sk-xxx"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    format: "responses"
  gemini:
    api_key: "AIza-xxx"
    base_url: "https://kaka.liugngg.top/v1"
    model: "gemini-1.5-flash"
    format: "chat_completions"
  claude:
    api_key: "sk-ant-xxx"
    base_url: "https://api.anthropic.com"
    model: "claude-3-5-sonnet-20241022"
    format: "chat_completions"

default_api: "openai"

settings:
  api_selection_timeout: 5  # 等待 5 秒后自动选择默认 API
```

### 交互示例
```
================================================================================
[*] 检测到以下可用的 LLM API 配置:
================================================================================
  1. openai          | Model: gpt-4o                        | URL: https://api.openai.com/v1 [默认]
  2. gemini          | Model: gemini-1.5-flash              | URL: https://kaka.liugngg.top/v1
  3. claude          | Model: claude-3-5-sonnet-20241022    | URL: https://api.anthropic.com
================================================================================
请选择 API (1-3) [默认: 1，5秒后自动选择]: 2
[*] 已选择: gemini
[*] 已选择 API 配置: [gemini] (Model: gemini-1.5-flash)
```

### 命令行优先级
1. **命令行参数** (`-a openai`) - 直接指定，跳过交互
2. **交互选择** - 用户在超时前输入序号或名称
3. **配置默认值** (`default_api`) - 超时后自动选择

### 跨平台支持
- **Windows**: 使用线程实现超时输入（因为 Windows 不支持 `select.select()` 用于标准输入）
- **Unix/Linux/macOS**: 使用 `select.select()` 实现高效超时

---

## 配置文件完整示例

```yaml
apis:
  openai:
    api_key: "YOUR_API_KEY"
    base_url: "https://api.openai.com/v1"
    model: "gpt-4o"
    format: "responses"
  gemini:
    api_key: "YOUR_API_KEY"
    base_url: "https://kaka.liugngg.top/v1"
    model: "gemini-1.5-flash"
    format: "chat_completions"

default_api: "openai"

settings:
  thread_num: 2
  batch_num: 10
  cache_dir: "./.cache_sub_trans"
  temperature: 0.3
  timeout: 45
  request_interval: 1
  fast_mode: true
  fast_mode_retry_threshold: 3      # 新增：批次重译阈值
  api_selection_timeout: 3          # 新增：API 选择超时
  merge_interval: 0.7
  max_duration: 8
  filter_non_semantic_vocalisation: true

prompt_file: "prompts.yaml"
```

---

## 使用建议

### 快速模式批次重译
- **推荐值**: 3-5 条
- **设置过低**: 过于敏感，频繁触发整批重译，降低效率
- **设置过高**: 容错性太强，漏翻较多时仍不重译

### API 选择超时
- **推荐值**: 3-5 秒
- **设置为 0**: 立即使用默认值，跳过交互（适合自动化脚本）
- **设置过长**: 增加等待时间，影响用户体验

---

## 技术细节

### 快速模式批次重译流程
```python
def _translate_chunk_with_retry(chunk):
    result = super()._translate_chunk_with_retry(chunk)
    if fast_mode:
        dropped = [item for item in result if should_drop(item)]
        if len(dropped) > threshold:
            # 清除缓存并重译整批
            cache.delete(cache_key)
            result = super()._translate_chunk_with_retry(chunk)
            # 重新过滤
            dropped = [item for item in result if should_drop(item)]
        # 返回过滤后的结果（移除漏翻条目）
        return [item for item in result if not should_drop(item)]
```

### API 选择超时实现
```python
# Windows: 使用线程避免阻塞
input_thread = threading.Thread(target=get_input, daemon=True)
input_thread.start()
input_thread.join(timeout=timeout_seconds)

# Unix: 使用 select 高效等待
ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
if ready:
    user_input = sys.stdin.readline()
```
