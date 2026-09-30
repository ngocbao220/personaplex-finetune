import unittest

from personaplex_finetuning.text_normalization import strip_vietnamese_diacritics


class VietnameseTextNormalizationTest(unittest.TestCase):
    def test_strips_vietnamese_marks_and_maps_d_stroke(self):
        self.assertEqual(
            strip_vietnamese_diacritics("Đặng Thị HƯƠNG, người dùng!"),
            "Dang Thi HUONG, nguoi dung!",
        )

    def test_handles_decomposed_unicode_without_changing_punctuation(self):
        self.assertEqual(
            strip_vietnamese_diacritics("a\u0300 Đ/đ"),
            "a D/d",
        )


if __name__ == "__main__":
    unittest.main()
