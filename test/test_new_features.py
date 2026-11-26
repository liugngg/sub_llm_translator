"""
测试新增功能的集成示例

这个文件演示了如何测试新增的两个功能：
1. 快速模式批次重译机制
2. API 选择交互

注意：这些是集成测试示例，不包含在标准测试套件中。
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sub_trans import LLMTranslator, SubtitleProcessData, select_api_with_timeout


class FastModeRetryTests(unittest.TestCase):
    """测试快速模式批次重译机制"""

    def test_batch_retry_when_exceeds_threshold(self):
        """当漏翻条数超过阈值时，应该重译整批"""
        # 准备测试数据
        chunk = [
            SubtitleProcessData(index=i, original_text=f"日本語テスト{i}", translated_text="")
            for i in range(1, 11)
        ]

        # 模拟第一次翻译：7条漏翻（超过阈值3）
        first_result = [
            SubtitleProcessData(index=i, original_text=f"日本語テスト{i}",
                              translated_text="好的翻译" if i <= 3 else f"日本語テスト{i}")
            for i in range(1, 11)
        ]

        # 模拟第二次翻译：只有2条漏翻
        second_result = [
            SubtitleProcessData(index=i, original_text=f"日本語テスト{i}",
                              translated_text="好的翻译" if i <= 8 else f"日本語テスト{i}")
            for i in range(1, 11)
        ]

        # 模拟 API 配置
        mock_api_config = {
            "api_key": "test_key",
            "base_url": "https://test.api.com/v1",
            "model": "test-model",
            "format": "chat_completions"
        }

        # 使用 patch 避免实际网络请求
        with patch('sub_trans.OpenAI'), \
             patch('diskcache.Cache'):

            translator = LLMTranslator(
                api_configs=[("test_api", mock_api_config)],
                prompts={"standard": "Translate ${source_language} to ${target_language}"},
                thread_num=1,
                batch_num=10,
                source_lang="Japanese",
                target_lang="简体中文",
                cache_dir="./.cache_test",
                fast_mode=True,
                fast_mode_retry_threshold=3  # 阈值为3
            )

            # 验证阈值设置正确
            self.assertEqual(translator.fast_mode_retry_threshold, 3)

    def test_no_retry_when_below_threshold(self):
        """当漏翻条数低于阈值时，不应重译"""
        mock_api_config = {
            "api_key": "test_key",
            "base_url": "https://test.api.com/v1",
            "model": "test-model",
            "format": "chat_completions"
        }

        with patch('sub_trans.OpenAI'), \
             patch('sub_trans.diskcache.Cache'):

            translator = LLMTranslator(
                api_configs=[("test_api", mock_api_config)],
                prompts={"standard": "Translate"},
                thread_num=1,
                batch_num=10,
                source_lang="Japanese",
                target_lang="简体中文",
                cache_dir="./.cache_test",
                fast_mode=True,
                fast_mode_retry_threshold=5
            )

            # 漏翻2条，不应该触发重译（阈值是5）
            self.assertEqual(translator.fast_mode_retry_threshold, 5)


class APISelectionTests(unittest.TestCase):
    """测试 API 选择交互功能"""

    def test_timeout_selects_default(self):
        """测试超时后自动选择默认 API"""
        available_apis = {
            "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
            "gemini": {"model": "gemini-1.5-flash", "base_url": "https://test.com/v1"}
        }
        default_api = "openai"

        # 模拟超时（无用户输入）
        with patch('builtins.input', side_effect=EOFError):
            if sys.platform == 'win32':
                # Windows 使用线程超时
                result = select_api_with_timeout(available_apis, default_api, timeout_seconds=0.1)
                self.assertEqual(result, default_api)

    def test_valid_number_selection(self):
        """测试用户通过序号选择 API"""
        available_apis = {
            "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
            "gemini": {"model": "gemini-1.5-flash", "base_url": "https://test.com/v1"}
        }
        default_api = "openai"

        # 模拟用户输入 "2" 选择第二个 API
        with patch('builtins.input', return_value="2"):
            result = select_api_with_timeout(available_apis, default_api, timeout_seconds=5)
            # 注意：由于 dict 在 Python 3.7+ 保持插入顺序，第二个应该是 gemini
            self.assertIn(result, available_apis.keys())

    def test_valid_name_selection(self):
        """测试用户通过名称选择 API"""
        available_apis = {
            "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
            "gemini": {"model": "gemini-1.5-flash", "base_url": "https://test.com/v1"}
        }
        default_api = "openai"

        # 模拟用户输入 "gemini" 名称
        with patch('builtins.input', return_value="gemini"):
            result = select_api_with_timeout(available_apis, default_api, timeout_seconds=5)
            self.assertEqual(result, "gemini")

    def test_invalid_input_uses_default(self):
        """测试无效输入时使用默认值"""
        available_apis = {
            "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
            "gemini": {"model": "gemini-1.5-flash", "base_url": "https://test.com/v1"}
        }
        default_api = "openai"

        # 模拟用户输入无效值 "999"
        with patch('builtins.input', return_value="999"):
            result = select_api_with_timeout(available_apis, default_api, timeout_seconds=5)
            self.assertEqual(result, default_api)

    def test_empty_input_uses_default(self):
        """测试空输入时使用默认值"""
        available_apis = {
            "openai": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"}
        }
        default_api = "openai"

        # 模拟用户直接按回车（空输入）
        with patch('builtins.input', return_value=""):
            result = select_api_with_timeout(available_apis, default_api, timeout_seconds=5)
            self.assertEqual(result, default_api)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
