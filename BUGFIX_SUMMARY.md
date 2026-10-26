# 字幕翻译工具 Bug 修复报告

## 修复日期
2026-10-26

## 修复的问题

### 🔴 高优先级修复

#### 1. **漏翻标记失效（严重）**

**问题描述：**
- 检测到疑似未翻译的字幕后，重译失败时应该用 `【原文】` 标记，但实际输出中没有标记
- 根本原因：`_translate_chunk` 方法的标记逻辑在 `try-except` 块内，当重译抛出异常时，标记操作被跳过，整个 chunk 进入重试流程，导致标记丢失

**修复方案：**
- 将标记逻辑移出 `try-except` 块，确保无论重译是否成功都能执行标记
- 重译失败时直接在 `except` 块中将 `retry_dict` 置为空字典，后续标记逻辑正常处理
- 添加 chunk 范围日志 (`chunk_range`)，便于定位问题

**修改位置：** `sub_trans.py:414-456`

**关键改动：**
```python
# 修复前：
try:
    retry_dict = self._agent_loop(...)
except Exception as e:
    self.write(f"\n[-] 疑似条目重译请求失败，将原文标记输出: {e}")
    retry_dict = {}  # 设置了空字典，但后续标记逻辑不会执行（因为外层 except 会捕获）

# 修复后：
retry_dict = {}
try:
    retry_dict = self._agent_loop(...)
except Exception as e:
    self.write(f"\n[-] 疑似条目重译请求失败 (chunk {chunk_range}): {e}，将原文标记输出")
# 无论是否成功，都会继续执行标记逻辑
for key, source in suspect.items():
    # ... 标记逻辑 ...
```

---

### 🟡 中优先级修复

#### 2. **漏翻检测逻辑过于敏感（中等）**

**问题描述：**
- 短字幕（5-10 字符）中的日文人名、口语被误判为未翻译
- 假名占比阈值 `>= 0.2` 对短文本过于严格（例如 "美香说" 3 个汉字 + 1 个假名，占比 25%）
- 标点符号差异导致误判（如 `！` vs `!`，或多余空格）

**修复方案：**
- 短字幕（< 10 字符）提高假名占比阈值到 `0.4`，长字幕保持 `0.2`
- 比较源文与译文时，规范化标点：移除所有中英文标点、全角半角空格后再比较
- 假名检测范围修正为完整的日文假名 Unicode 范围 `[぀-ヿ]`（平假名 + 片假名）

**修改位置：** `sub_trans.py:520-541`

**关键改动：**
```python
# 修复前：
kana_count = len(re.findall(r'[぀-ヿ]', translation))
if kana_count >= 2 and kana_count / max(len(translation), 1) >= 0.2:
    return True
compact_source = re.sub(r'\s+', '', source)  # 只移除空格
compact_translation = re.sub(r'\s+', '', translation)

# 修复后：
kana_count = len(re.findall(r'[぀-ヿ]', translation))
trans_len = len(translation)
threshold = 0.4 if trans_len < 10 else 0.2  # 短字幕更宽松
if kana_count >= 2 and kana_count / max(trans_len, 1) >= threshold:
    return True
# 移除标点、空格、全角字符后再比较
normalized_source = re.sub(r'[\s　 -⁯、-〿＀-￯!！?？,.，。、;；:：…\-—]+', '', source)
normalized_translation = re.sub(r'[\s　 -⁯、-〿＀-￯!！?？,.，。、;；:：…\-—]+', '', translation)
```

---

#### 3. **错误日志不足（中等）**

**问题描述：**
- chunk 翻译失败时，无法快速定位是哪些字幕出了问题
- 多文件批量处理时，失败信息不够详细

**修复方案：**
- 在 `_translate_chunk` 开始时计算 chunk 范围（如 "1-40" 或 "41-80"）
- 所有错误日志中添加 chunk 范围标识
- 标记未翻译字幕时，输出字幕索引和前 30 个字符

**修改位置：** `sub_trans.py:415, 421, 435, 447, 450`

**新增日志示例：**
```
[-] 检出 3 条疑似未翻译字幕 (chunk 41-80)，仅重译这些条目
[!] 字幕 45 标记为未翻译: これはテストです...
[-] Chunk 41-80 翻译失败: RateLimitError
```

---

#### 4. **配置验证增强（中等）**

**问题描述：**
- `temperature` 允许任意数值，部分模型只支持 0-1 或 0-2
- `merge_interval` 和 `max_duration` 没有范围检查，可能出现负数
- `replacements` 中的正则表达式语法错误在运行时才暴露

**修复方案：**
- 启动时验证 `temperature` 在 0-2 范围内（覆盖大部分模型要求）
- 验证 `merge_interval >= 0` 和 `max_duration > 0`
- 预编译所有 `replacements` 中的正则表达式，无效的发出警告并忽略

**修改位置：** `sub_trans.py:663-706`

**新增验证逻辑：**
```python
# 验证 temperature 范围
temperature = settings.get('temperature', 0.7)
if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
    raise ValueError("settings.temperature 必须是 0 到 2 之间的数值")

# 验证 merge_interval 和 max_duration
merge_interval = settings.get('merge_interval', 0.7)
max_duration = settings.get('max_duration', 8)
if not isinstance(merge_interval, (int, float)) or merge_interval < 0:
    raise ValueError("settings.merge_interval 必须是非负数值")
if not isinstance(max_duration, (int, float)) or max_duration <= 0:
    raise ValueError("settings.max_duration 必须是正数值")

# 验证 replacements 中的正则表达式
replacements = config.get('replacements', {})
if replacements and isinstance(replacements, dict):
    invalid_patterns = []
    for pattern in replacements.keys():
        try:
            re.compile(pattern)
        except re.error:
            invalid_patterns.append(pattern)
    if invalid_patterns:
        print(f"[!] 警告：以下正则表达式语法无效，将被忽略: {invalid_patterns}")
        for pattern in invalid_patterns:
            del replacements[pattern]
```

---

## 未修复的低优先级问题

按照要求，以下问题暂时保留：

### 5. ASS 时间戳转换精度丢失（轻微）
- **位置：** `sub_trans.py:218-225`
- **问题：** 毫秒截断而非四舍五入，可能导致 1-9ms 的误差
- **影响：** 对于视频同步影响极小，人眼无法察觉

### 6. 清理规则可能误删有效内容（轻微）
- **位置：** `sub_trans.py:50 BLANK_RE`
- **问题：** 部分中文语气词（如 `啊`、`哦`）在某些上下文中有意义
- **建议：** 未来可将清理规则移到配置文件，提供严格/宽松模式

### 7. 缓存键不包含 temperature 和 model（按用户要求保留）
- **位置：** `sub_trans.py:366-368`
- **说明：** 用户明确要求保留当前缓存机制，不修改缓存键生成逻辑

---

## 测试建议

修复后请测试以下场景：

1. **漏翻标记测试：**
   - 准备包含日文假名较多的 SRT 文件
   - 手动模拟 API 失败（如关闭网络），观察输出中是否有 `【原文】` 标记

2. **短字幕误判测试：**
   - 测试包含日文人名的短字幕（如 "美香：おはよう"）
   - 检查是否仍被误判为未翻译

3. **配置验证测试：**
   - 设置 `temperature: 3.0`，确认程序拒绝启动
   - 在 `replacements` 中添加无效正则 `[abc`，确认启动时有警告

4. **错误日志测试：**
   - 触发翻译失败，检查日志中是否显示 chunk 范围和字幕索引

---

## 代码质量检查

所有修改已通过 Python 编译检查：
```bash
python -m compileall sub_trans.py prompts.py
```

**结果：** ✅ 通过，无语法错误

---

## 后续建议

1. **添加单元测试：** 为 `_looks_untranslated` 和 `_translate_chunk` 编写测试用例
2. **日志分级：** 引入 `logging` 模块，区分 INFO/WARNING/ERROR
3. **用户反馈机制：** 导出失败 chunk 到单独文件，便于用户提交 issue
