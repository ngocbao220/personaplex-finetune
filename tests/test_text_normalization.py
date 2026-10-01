import unittest

from personaplex_finetuning.text_normalization import (
    decode_vietnamese_telex,
    encode_vietnamese_telex,
    strip_vietnamese_diacritics,
)


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

    def test_telex_round_trip_and_keyboard_forms(self):
        examples = {
            "tương": "tuowng",
            "tiếng Việt": "tieengs Vieetj",
            "Đặng Thị HƯƠNG, người dùng!": "Ddawngj Thij HUOWNG, nguwowif dungf!",
            "hòa": "hoaf",
        }
        for unicode_text, telex_text in examples.items():
            with self.subTest(unicode_text=unicode_text):
                self.assertEqual(encode_vietnamese_telex(unicode_text), telex_text)
                self.assertEqual(decode_vietnamese_telex(telex_text), unicode_text)

    def test_telex_preserves_ascii_and_decomposed_unicode(self):
        self.assertEqual(encode_vietnamese_telex("hello, tuowng!"), "hello, tuowng!")
        self.assertEqual(decode_vietnamese_telex("tieengs Vieetj"), "tiếng Việt")
        self.assertEqual(encode_vietnamese_telex("a\u0300"), "af")

    def test_telex_handles_all_tones_and_vietnamese_letter_shapes(self):
        examples = {
            "mà": "maf", "má": "mas", "mả": "mar", "mã": "max", "mạ": "maj",
            "sáng": "sangs", "bão": "baox", "kế": "kees", "tô": "too",
            "mơ": "mow", "tư": "tuw", "đỗ": "ddoox", "MÁ": "MAS",
            "quá": "quas", "giá": "gias",
        }
        for unicode_text, telex_text in examples.items():
            with self.subTest(unicode_text=unicode_text):
                self.assertEqual(encode_vietnamese_telex(unicode_text), telex_text)
                self.assertEqual(decode_vietnamese_telex(telex_text), unicode_text)


if __name__ == "__main__":
    unittest.main()
