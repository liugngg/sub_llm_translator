import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sub_trans import (
    CensoredTranslationError,
    FatalTranslationError,
    LLMTranslator,
    SubtitleProcessData,
    TranslationError,
)


class TranslationWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.translator = LLMTranslator(
            api_configs=[("test", {
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "model": "test-model",
            })],
            prompts={"standard": "Translate ${source_language} to ${target_language}. Return JSON."},
            thread_num=1,
            batch_num=10,
            source_lang="Japanese",
            target_lang="简体中文",
            cache_dir=temp_dir.name,
        )
        self.addCleanup(self.translator.cache.close)
        self.addCleanup(self.translator.client.close)
        self.addCleanup(self.translator.stop)
        self.translator.write = Mock()

    def response(self, content, finish_reason="stop", refusal=None):
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content, refusal=refusal),
        )])

    def test_ordinary_dialogue_is_not_a_refusal(self):
        for dialogue in ("抱歉", "我不能走", "色情", "できません", "I'm sorry", "违反约定"):
            with self.subTest(dialogue=dialogue):
                self.assertFalse(self.translator._looks_like_provider_refusal(dialogue))
        for refusal in ("I cannot translate this content.", "抱歉，我无法翻译这些内容。",
                        "この内容を翻訳できません"):
            with self.subTest(refusal=refusal):
                self.assertTrue(self.translator._looks_like_provider_refusal(refusal))

    def test_json_with_ordinary_dialogue_is_accepted(self):
        payload = {"1": "抱歉，我不能走。", "2": "违反约定"}
        with patch.object(self.translator.client.chat.completions, "create",
                          return_value=self.response(json.dumps(payload))) as request:
            self.assertEqual(self.translator._agent_loop("Return JSON", {"1": "a", "2": "b"}), payload)
        self.assertEqual(request.call_count, 1)

    def test_explicit_chat_refusal_is_not_reported_as_empty_content(self):
        for response in (self.response(None, "content_filter"),
                         self.response(None, refusal="Cannot translate this content")):
            with self.subTest(response=response):
                with patch.object(self.translator.client.chat.completions, "create", return_value=response):
                    with self.assertRaises(CensoredTranslationError):
                        self.translator._agent_loop("Return JSON", {"1": "a"})

    def test_explicit_responses_refusal_is_detected(self):
        self.translator.endpoints[0]["format"] = "responses"
        response = SimpleNamespace(output_text="", output=[SimpleNamespace(
            content=[SimpleNamespace(type="refusal")])])
        with patch.object(self.translator.client.responses, "create", return_value=response):
            with self.assertRaises(CensoredTranslationError):
                self.translator._agent_loop("Return JSON", {"1": "a"})

    def test_partial_refusal_keeps_other_translations_without_retry(self):
        chunk = [SubtitleProcessData(1, "こんにちは"), SubtitleProcessData(2, "さようなら")]
        with patch.object(self.translator, "_agent_loop", return_value={
            "1": "你好", "2": "I cannot translate this content."
        }) as request:
            result = self.translator._translate_chunk(chunk)
        self.assertEqual([d.translated_text for d in result], ["你好", "【云端审查】さようなら"])
        self.assertEqual(request.call_count, 1)

    def test_refused_retranslation_keeps_successful_entries(self):
        chunk = [SubtitleProcessData(1, "こんにちは"), SubtitleProcessData(2, "さようなら")]
        with patch.object(self.translator, "_agent_loop", side_effect=[
            {"1": "你好", "2": "さようなら"}, CensoredTranslationError("blocked")
        ]) as request:
            result = self.translator._translate_chunk(chunk)
        self.assertEqual([d.translated_text for d in result], ["你好", "【云端审查】さようなら"])
        self.assertEqual(request.call_count, 2)

    def test_fatal_retranslation_error_propagates(self):
        with patch.object(self.translator, "_agent_loop", side_effect=[
            {"1": "さようなら"}, FatalTranslationError("authentication failed")
        ]):
            with self.assertRaises(FatalTranslationError):
                self.translator._translate_chunk([SubtitleProcessData(1, "さようなら")])

    def test_split_preserves_entries_and_caches_parent_for_one_hour(self):
        def translate(chunk):
            if any(d.index == 2 for d in chunk):
                raise CensoredTranslationError("blocked")
            for data in chunk:
                data.translated_text = "译文"
            return chunk

        chunk = [SubtitleProcessData(i, f"source {i}") for i in range(1, 4)]
        key = self.translator._generate_chunk_cache_key(chunk)
        with patch.object(self.translator, "_translate_chunk", side_effect=translate) as request:
            result = self.translator._process_chunk_with_retry(chunk)
            calls = request.call_count
            cached = self.translator._process_chunk_with_retry(
                [SubtitleProcessData(i, f"source {i}") for i in range(1, 4)])
            self.assertEqual(request.call_count, calls)
        self.assertEqual([d.index for d in result], [1, 2, 3])
        self.assertEqual([d.translated_text for d in cached], ["译文", "【云端审查】source 2", "译文"])
        self.assertIsNotNone(self.translator.cache.get(key))

    def test_failure_retries_three_times_without_caching(self):
        chunk = [SubtitleProcessData(1, "source")]
        key = self.translator._generate_chunk_cache_key(chunk)
        with patch.object(self.translator, "_translate_chunk", side_effect=TranslationError("temporary")) as request:
            with patch("sub_trans.time.sleep") as sleep:
                result = self.translator._process_chunk_with_retry(chunk)
        self.assertEqual(request.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertEqual(result[0].translated_text, "【翻译失败】source")
        self.assertIsNone(self.translator.cache.get(key))

    def test_fatal_error_is_not_retried(self):
        with patch.object(self.translator, "_translate_chunk", side_effect=FatalTranslationError("bad key")) as request:
            with self.assertRaises(FatalTranslationError):
                self.translator._process_chunk_with_retry([SubtitleProcessData(1, "source")])
        self.assertEqual(request.call_count, 1)

    def test_cache_key_tracks_prompt_model_and_ignores_translation_state(self):
        chunk = [SubtitleProcessData(1, "source")]
        first = self.translator._generate_chunk_cache_key(chunk)
        chunk[0].translated_text = "译文"
        self.assertEqual(first, self.translator._generate_chunk_cache_key(chunk))
        self.translator.prompts["standard"] += " Updated."
        second = self.translator._generate_chunk_cache_key(chunk)
        self.assertNotEqual(first, second)
        self.translator.endpoints[0]["model"] = "other-model"
        self.assertNotEqual(second, self.translator._generate_chunk_cache_key(chunk))


class RequestIntervalAndFailoverTests(unittest.TestCase):
    def make_translator(self, endpoints=1, request_interval=0.0):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        configs = [
            (f"api{i + 1}", {
                "api_key": f"key{i + 1}",
                "base_url": f"https://api{i + 1}.invalid/v1",
                "model": f"model{i + 1}",
            })
            for i in range(endpoints)
        ]
        translator = LLMTranslator(
            api_configs=configs,
            prompts={"standard": "Translate ${source_language} to ${target_language}. Return JSON."},
            thread_num=1,
            batch_num=10,
            source_lang="Japanese",
            target_lang="简体中文",
            cache_dir=temp_dir.name,
            request_interval=request_interval,
        )
        self.addCleanup(translator.cache.close)
        for endpoint in translator.endpoints:
            self.addCleanup(endpoint["client"].close)
        self.addCleanup(translator.stop)
        translator.write = Mock()
        return translator

    def test_throttle_waits_between_requests(self):
        translator = self.make_translator(request_interval=1.0)
        with patch("sub_trans.time.sleep") as sleep, patch("sub_trans.time.monotonic") as clock:
            clock.side_effect = [100.0, 100.2, 100.2]
            translator._throttle()
            translator._throttle()
        self.assertEqual(len(sleep.call_args_list), 1)
        self.assertAlmostEqual(sleep.call_args_list[0].args[0], 0.8)
        self.assertAlmostEqual(translator._next_request_time, 101.2)

    def test_zero_interval_does_not_sleep(self):
        translator = self.make_translator(request_interval=0)
        with patch("sub_trans.time.sleep") as sleep:
            translator._throttle()
            translator._throttle()
        sleep.assert_not_called()

    def test_censored_batch_fails_over_to_next_service(self):
        translator = self.make_translator(endpoints=2)
        attempts = []

        def translate(chunk):
            attempts.append(translator._endpoint["name"])
            if translator._endpoint["name"] == "api1":
                raise CensoredTranslationError("blocked")
            for data in chunk:
                data.translated_text = "译文"
            return chunk

        with patch.object(translator, "_translate_chunk", side_effect=translate):
            result = translator._process_chunk_with_retry([SubtitleProcessData(1, "こんにちは")])
        self.assertEqual(attempts, ["api1", "api2"])
        self.assertEqual(result[0].translated_text, "译文")
        # 成功端点成为其他线程的优先起点
        self.assertEqual(translator._preferred_endpoint(), 1)

    def test_auth_failure_fails_over_to_next_service(self):
        translator = self.make_translator(endpoints=2)

        def translate(chunk):
            if translator._endpoint["name"] == "api1":
                raise FatalTranslationError("bad key")
            chunk[0].translated_text = "译文"
            return chunk

        with patch.object(translator, "_translate_chunk", side_effect=translate):
            result = translator._process_chunk_with_retry([SubtitleProcessData(1, "こんにちは")])
        self.assertEqual(result[0].translated_text, "译文")

    def test_all_services_censored_marks_single_entry(self):
        translator = self.make_translator(endpoints=2)
        with patch.object(translator, "_translate_chunk",
                          side_effect=CensoredTranslationError("blocked")) as request:
            result = translator._process_chunk_with_retry([SubtitleProcessData(1, "こんにちは")])
        self.assertEqual(request.call_count, 2)
        self.assertEqual(result[0].translated_text, "【云端审查】こんにちは")

    def test_all_services_censored_splits_multi_entry_batch(self):
        translator = self.make_translator(endpoints=2)
        seen = []

        def translate(chunk):
            seen.append(len(chunk))
            raise CensoredTranslationError("blocked")

        with patch.object(translator, "_translate_chunk", side_effect=translate):
            result = translator._process_chunk_with_retry(
                [SubtitleProcessData(i, f"source{i}") for i in (1, 2)])
        self.assertEqual(seen[:2], [2, 2])
        self.assertEqual([d.translated_text for d in result],
                         ["【云端审查】source1", "【云端审查】source2"])

    def test_single_service_fatal_error_still_raises(self):
        translator = self.make_translator(endpoints=1)
        with patch.object(translator, "_translate_chunk", side_effect=FatalTranslationError("bad key")):
            with self.assertRaises(FatalTranslationError):
                translator._process_chunk_with_retry([SubtitleProcessData(1, "source")])

    def test_cache_keys_differ_between_services(self):
        translator = self.make_translator(endpoints=2)
        chunk = [SubtitleProcessData(1, "source")]
        translator._activate_endpoint(0)
        first = translator._generate_chunk_cache_key(chunk)
        translator._activate_endpoint(1)
        self.assertNotEqual(first, translator._generate_chunk_cache_key(chunk))


if __name__ == "__main__":
    unittest.main()
