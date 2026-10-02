from parser import build_full_content, clean_html_text, format_date, parse_legislation_data

DETAIL = {
    "serial": "aGx3rOE4CvQArw0oIGZqHA",
    "name": "اللائحة التنفيذية لنظام الإفلاس",
    "legalType": "لائحة",
    "legalStatueName": "ساري",
    "classificationName": "الوثائق الأخرى",
    "summary": "<p>تضمنت اللائحة&nbsp;: أحكام عامة</p>",
    "issuanceDate": "1439-12-24T00:00:00",
    "publishDate": "1440-01-04T00:00:00",
    "statuteStructure": [
        {
            "type": 3, "name": "أحكام عامة", "sequence": "الفصل الأول", "text": None, "decree": None,
            "items": [
                {"type": 1, "sequence": "المادة الأولى", "name": "المادة الأولى",
                 "decree": "قرار مجلس الوزراء رقم (171)",
                 "text": "<p>١ - يكون للألفاظ والمصطلحات</p><p>٢ - يقصد بالألفاظ</p>", "items": []},
                {"type": 1, "sequence": "المادة الثانية", "name": None, "decree": None,
                 "text": "<div>تسري أحكام اللائحة</div>", "items": None},
            ],
        }
    ],
}


def test_parse_detail_with_nested_chapters_has_content():
    leg = parse_legislation_data({"serial": "aGx3rOE4CvQArw0oIGZqHA"}, DETAIL)
    assert leg.title == "اللائحة التنفيذية لنظام الإفلاس"
    assert leg.content and "المادة الأولى" in leg.content and "تسري أحكام اللائحة" in leg.content
    assert "## الفصل الأول - أحكام عامة" in leg.content
    assert leg.decree == "قرار مجلس الوزراء رقم (171)"
    assert leg.issue_date == "1439-12-24"
    assert leg.description == "تضمنت اللائحة : أحكام عامة"
    assert leg.source_url.endswith("/aGx3rOE4CvQArw0oIGZqHA")


def test_parse_without_detail_falls_back_to_search_item():
    leg = parse_legislation_data({"serial": " X1 ", "statuteName": "نظام العمل", "legalType": "نظام"}, None)
    assert leg.serial == "X1" and leg.title == "نظام العمل" and leg.legislation_type == "نظام"
    assert leg.content is None


def test_parse_tolerates_non_string_and_malformed_values():
    detail = {"serial": 123, "name": "نظام", "statuteStructure": [None, {"sequence": 5, "text": None, "items": ["bad"]}]}
    leg = parse_legislation_data({}, detail)
    assert leg.serial == "123"
    assert leg.content == "## 5"


def test_clean_html_text_preserves_arabic_and_paragraphs():
    assert clean_html_text("<p>المادة&nbsp;الأولى</p><br><p> نص </p>") == "المادة الأولى\nنص"
    assert clean_html_text(None) == ""


def test_format_date():
    assert format_date("1440-09-21T00:00:00") == "1440-09-21"
    assert format_date("1440/9/1") == "1440-9-1"
    assert format_date("") is None and format_date(None) is None


def test_build_full_content_empty():
    assert build_full_content([]) is None
    assert build_full_content(None) is None
