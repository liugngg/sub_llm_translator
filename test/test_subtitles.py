import tempfile
import unittest
from pathlib import Path

from sub_trans import ASRData, ASRDataSeg


def segment(index, text, start, end, translated=""):
    return ASRDataSeg(text, start, end, index, translated)


class SubtitleProcessingTests(unittest.TestCase):
    def test_parse_utf8_japanese_srt_and_clean_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "japanese.srt"
            source.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nあああ\n\n"
                "2\n00:00:02,100 --> 00:00:03,000\nえっ\n\n"
                "3\n00:00:03,100 --> 00:00:04,000\nうん\n\n"
                "4\n00:00:04,100 --> 00:00:05,000\n？！\n",
                encoding="utf-8",
            )
            data = ASRData.from_srt(str(source))

        self.assertEqual([seg.text for seg in data.segments], ["あああ", "えっ", "うん", "？！"])
        data.srt_clean()
        self.assertEqual([seg.text for seg in data.segments], ["あああ", "えっ", "うん"])
        self.assertEqual([seg.index for seg in data.segments], [1, 2, 3])
        self.assertEqual(data.clean_line("こんにちはこんにちは"), "こんにちは")

    def test_configured_replacements_apply(self):
        data = ASRData([])
        data.replace_dic = {"猫": "ねこ", "不要": ""}
        self.assertEqual(data.clean_line("猫がいる"), "ねこがいる")
        self.assertEqual(data.clean_line("これは不要です"), "これはです")

    def test_merge_identical_non_overlapping_within_interval_and_duration_cap(self):
        data = ASRData([
            segment(1, "同じ", "00:00:00,000", "00:00:01,000"),
            segment(2, "同じ", "00:00:01,500", "00:00:02,000"),
            segment(3, "别", "00:00:02,100", "00:00:12,200"),
        ])
        data.merge_interval = 0.7
        data.max_duration = 8
        data.srt_merge()
        self.assertEqual(len(data.segments), 2)
        self.assertEqual(data.segments[0].end_time, "00:00:02,000")
        self.assertEqual(data.segments[1].end_time, "00:00:10,100")
        self.assertEqual([seg.index for seg in data.segments], [1, 2])

    def test_overlaps_and_different_text_are_not_merged(self):
        data = ASRData([
            segment(1, "同じ", "00:00:00,000", "00:00:01,000"),
            segment(2, "同じ", "00:00:00,900", "00:00:01,500"),
            segment(3, "不同", "00:00:01,600", "00:00:02,000"),
        ])
        data.srt_merge()
        self.assertEqual(len(data.segments), 3)

    def test_millisecond_rounding_carries_into_next_second(self):
        self.assertEqual(ASRData.seconds_to_time(1.9996), "00:00:02,000")

    def test_srt_exports_chinese_and_bilingual_with_original_timeline(self):
        data = ASRData([
            segment(8, "こんにちは", "00:00:01,234", "00:00:02,345", "你好"),
            segment(9, "さようなら", "00:00:03,000", "00:00:04,000", "再见"),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            chinese = Path(temp_dir) / "zh.srt"
            bilingual = Path(temp_dir) / "bi.srt"
            data.to_srt(str(chinese))
            data.to_srt(str(bilingual), bilingual=True)
            chinese_text = chinese.read_text(encoding="utf-8-sig")
            bilingual_text = bilingual.read_text(encoding="utf-8-sig")

        self.assertIn("1\n00:00:01,234 --> 00:00:02,345\n你好\n", chinese_text)
        self.assertIn("2\n00:00:03,000 --> 00:00:04,000\n再见\n", chinese_text)
        self.assertIn("1\n00:00:01,234 --> 00:00:02,345\n你好\nこんにちは\n", bilingual_text)
        self.assertIn("2\n00:00:03,000 --> 00:00:04,000\n再见\nさようなら\n", bilingual_text)

    def test_ass_export_has_ass_timestamps_and_bilingual_line_break(self):
        data = ASRData([
            segment(1, "こんにちは", "00:00:01,234", "00:00:02,345", "你好"),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.ass"
            data.to_ass(str(output), bilingual=True)
            contents = output.read_text(encoding="utf-8-sig")

        self.assertIn("Dialogue: 0,0:00:01.23,0:00:02.34,Default,,0,0,0,,你好\\Nこんにちは", contents)

    def test_failure_marks_survive_output_cleanup(self):
        data = ASRData([
            segment(1, "原文", "00:00:01,000", "00:00:02,000", "【云端审查】原文"),
            segment(2, "失败", "00:00:03,000", "00:00:04,000", "【翻译失败】失败"),
        ])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "marked.srt"
            data.to_srt(str(output))
            contents = output.read_text(encoding="utf-8-sig")

        self.assertIn("【云端审查】原文", contents)
        self.assertIn("【翻译失败】失败", contents)


if __name__ == "__main__":
    unittest.main()
