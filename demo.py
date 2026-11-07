"""Replay invented provider responses through the real processing pipeline."""

from io import BytesIO
from unittest.mock import patch

from PIL import Image

import api

CONVERSATION = "Alex: Want to visit the exhibition on Saturday?\nSam: Sounds good. What time?"
SUGGESTIONS = "Friendly: Shall we meet at two?\nFlexible: What time works for you?"


def run_demo() -> str:
    image = BytesIO()
    Image.new("RGB", (320, 160), "white").save(image, "PNG")
    calls = []

    def fixture_post(url, body):
        calls.append(url)
        if url == api.OCR_URL:
            assert body["mimeType"] == "image/jpeg"
            return {"textAnnotation": {"fullText": CONVERSATION}}
        assert CONVERSATION in body["messages"][1]["text"]
        return {"result": {"alternatives": [{"message": {"text": SUGGESTIONS}}]}}

    with patch.object(api, "_post", side_effect=fixture_post):
        result = api.suggest_from_image(image.getvalue())
    assert calls == [api.OCR_URL, api.GPT_URL]
    assert result == SUGGESTIONS
    return result


if __name__ == "__main__":
    print("Offline fixture replay: OCR and reply generation are simulated. No requests sent.")
    print(run_demo())
