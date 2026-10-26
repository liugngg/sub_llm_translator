#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证 Bug 修复
"""

import re
from typing import Any


def test_looks_untranslated():
    """测试漏翻检测逻辑"""

    def _looks_untranslated(source: str, translation: Any) -> bool:
        """复制修复后的检测逻辑"""
        if not isinstance(translation, str) or not translation.strip():
            return True

        # 日译中检测
        kana_count = len(re.findall(r'[\u3040-\u30ff]', translation))
        trans_len = len(translation)
        threshold = 0.4 if trans_len < 10 else 0.2
        if kana_count >= 2 and kana_count / max(trans_len, 1) >= threshold:
            return True

        # 规范化标点后比较
        normalized_source = re.sub(r'[\s\u3000\u2000-\u206f\u3001-\u303f\uff00-\uffef!\uff01?\uff1f,.\uff0c\u3002\u3001;\uff1b:\uff1a\u2026\-\u2014]+', '', source)
        normalized_translation = re.sub(r'[\s\u3000\u2000-\u206f\u3001-\u303f\uff00-\uffef!\uff01?\uff1f,.\uff0c\u3002\u3001;\uff1b:\uff1a\u2026\-\u2014]+', '', translation)
        if len(normalized_source) >= 4 and normalized_source == normalized_translation:
            return True

        return False

    # 测试用例
    test_cases = [
        # (source, translation, expected_result, description)
        ("こんにちは", "你好", False, "正常翻译"),
        ("こんにちは", "こんにちは", True, "完全未翻译"),
        ("美香：おはよう", "美香：早上好", False, "短字幕，人名+正常翻译（修复前可能误判）"),
        ("これはテストです", "これはテストです", True, "假名过多未翻译"),
        ("こんにちは！", "你好！", False, "标点差异已规范化（修复前可能误判）"),
        ("こんにちは", "你好!!", False, "标点不同但内容已翻译"),
        ("田中さん", "田中", False, "短字幕人名（修复前可能误判）"),
        ("はい", "是", False, "单字翻译"),
        ("あああああ", "啊啊啊啊啊", False, "语气词翻译"),
        ("", "", True, "空字符串"),
        ("テスト", "テスト", True, "短假名未翻译"),
    ]

    print("=" * 80)
    print("漏翻检测逻辑测试")
    print("=" * 80)

    passed = 0
    failed = 0

    for source, translation, expected, description in test_cases:
        result = _looks_untranslated(source, translation)
        status = "✓" if result == expected else "✗"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status} {description}")
        print(f"  原文: {source}")
        print(f"  译文: {translation}")
        print(f"  期望: {'疑似未翻译' if expected else '正常'} | 实际: {'疑似未翻译' if result else '正常'}")
        print()

    print("=" * 80)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 80)

    return failed == 0


def test_config_validation():
    """测试配置验证逻辑"""

    print("\n" + "=" * 80)
    print("配置验证测试")
    print("=" * 80)

    # 测试 temperature 验证
    test_temps = [
        (0.7, True, "正常温度"),
        (0, True, "最小值"),
        (2, True, "最大值"),
        (-0.1, False, "负数"),
        (2.5, False, "超出范围"),
        ("0.7", False, "字符串类型"),
    ]

    for temp, expected, desc in test_temps:
        try:
            is_valid = isinstance(temp, (int, float)) and 0 <= temp <= 2
            status = "✓" if is_valid == expected else "✗"
            print(f"{status} temperature={temp} ({desc}): {'有效' if is_valid else '无效'}")
        except Exception as e:
            print(f"✗ temperature={temp} 验证异常: {e}")

    print()

    # 测试正则表达式验证
    test_patterns = [
        (r'\d+', True, "正常正则"),
        (r'[abc]', True, "字符类"),
        (r'[abc', False, "未闭合字符类"),
        (r'(?P<name>\w+)', True, "命名组"),
        (r'(?P<name>\w+', False, "未闭合组"),
    ]

    for pattern, expected, desc in test_patterns:
        try:
            re.compile(pattern)
            is_valid = True
        except re.error:
            is_valid = False

        status = "✓" if is_valid == expected else "✗"
        print(f"{status} 正则 '{pattern}' ({desc}): {'有效' if is_valid else '无效'}")

    print("=" * 80)


if __name__ == "__main__":
    all_passed = test_looks_untranslated()
    test_config_validation()

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败，请检查实现")
    print("=" * 80)
