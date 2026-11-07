import asyncio
import json
import os
import socket
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

import api
import bot
import demo


def image_bytes(mode="RGB", fmt="PNG"):
    stream = BytesIO()
    Image.new(mode, (80, 40)).save(stream, fmt)
    return stream.getvalue()


class PipelineTests(unittest.TestCase):
    def test_image_is_always_jpeg_and_bounded(self):
        for mode in ("RGB", "RGBA", "L"):
            with self.subTest(mode=mode):
                data = api.compress_image_if_needed(image_bytes(mode))
                with Image.open(BytesIO(data)) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.mode, "RGB")
                    self.assertFalse(image.getexif())
                self.assertLessEqual(len(data), 0.9 * 1024 * 1024)

    def test_image_size_and_content_rejected(self):
        for data in (b"", b"not an image", b"x" * (api.MAX_INPUT_BYTES + 1)):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                api.compress_image_if_needed(data)
        with patch.object(api, "MAX_IMAGE_PIXELS", 10), self.assertRaises(ValueError):
            api.compress_image_if_needed(image_bytes())

    def test_gpt_malformed_responses(self):
        for data in (None, [], "bad json", {}, {"result": None},
                     {"result": {"alternatives": []}},
                     {"result": {"alternatives": [{"message": {"text": None}}]}}):
            with self.subTest(data=data):
                self.assertEqual(api.extract_gpt_response(data), "")

    def test_gpt_accepts_both_response_envelopes(self):
        data = {"alternatives": [{"message": {"text": " reply "}}]}
        for value in (data, {"result": data}, json.dumps({"result": data})):
            self.assertEqual(api.extract_gpt_response(value), "reply")

    def test_offline_replay_has_no_network_even_with_live_environment(self):
        with patch.dict(os.environ, {"LIVE_INTEGRATIONS_ENABLED": "true"}), \
                patch.object(socket.socket, "connect", side_effect=AssertionError("network")), \
                patch.object(api.requests, "post", side_effect=AssertionError("network")):
            self.assertEqual(demo.run_demo(), demo.SUGGESTIONS)

    def test_live_opt_in_and_credentials_checked_before_network(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(api.requests, "post") as post:
            with self.assertRaises(api.ServiceError):
                api._post(api.OCR_URL, {})
            os.environ["LIVE_INTEGRATIONS_ENABLED"] = "true"
            with self.assertRaises(api.ServiceError):
                api._post(api.OCR_URL, {})
            post.assert_not_called()

    def test_provider_transport_is_bounded_and_does_not_follow_redirects(self):
        env = {"LIVE_INTEGRATIONS_ENABLED": "true", "FOLDER_ID": "demo-folder",
               "YANDEX_AI_API": "test-placeholder"}
        response = Mock(status_code=200)
        response.json.return_value = {"textAnnotation": {"fullText": "hello"}}
        with patch.dict(os.environ, env), patch.object(api.requests, "post", return_value=response) as post:
            self.assertEqual(api.extract_text_with_vision_ocr(b"jpeg"), "hello")
            kwargs = post.call_args.kwargs
            self.assertEqual(kwargs["timeout"], (5, 40))
            self.assertFalse(kwargs["allow_redirects"])
            self.assertEqual(kwargs["json"]["mimeType"], "image/jpeg")

    def test_provider_errors_do_not_include_response_body(self):
        env = {"LIVE_INTEGRATIONS_ENABLED": "true", "FOLDER_ID": "demo-folder",
               "YANDEX_AI_API": "test-placeholder"}
        response = Mock(status_code=403, text="private conversation and credentials")
        with patch.dict(os.environ, env), patch.object(api.requests, "post", return_value=response):
            with self.assertRaises(api.ServiceError) as error:
                api._post(api.OCR_URL, {})
            self.assertNotIn("private conversation", str(error.exception))

    def test_ocr_empty_invalid_and_oversized_text_rejected(self):
        for response in ({}, {"result": None}, {"textAnnotation": {"fullText": []}},
                         {"textAnnotation": {"fullText": "x" * (api.MAX_TEXT_CHARS + 1)}}):
            with self.subTest(response=str(response)[:60]), \
                    patch.object(api, "_post", return_value=response), \
                    self.assertRaises((api.ServiceError, ValueError)):
                api.extract_text_with_vision_ocr(b"jpeg")

    def test_pipeline_stops_on_ocr_failure(self):
        with patch.object(api, "extract_text_with_vision_ocr", side_effect=api.ServiceError("OCR failed")), \
                patch.object(api, "process_text_with_yandexgpt") as gpt:
            with self.assertRaises(api.ServiceError):
                api.suggest_from_image(image_bytes())
            gpt.assert_not_called()

    def test_conversation_limits_checked_before_request(self):
        for text in ("", " ", "x" * (api.MAX_TEXT_CHARS + 1)):
            with patch.object(api, "_post") as post, self.assertRaises(ValueError):
                api.process_text_with_yandexgpt(text)
            post.assert_not_called()

    def test_access_requires_private_allowlisted_user(self):
        users = bot.allowed_user_ids("1001, 1002")
        message = SimpleNamespace(chat=SimpleNamespace(type="private"), from_user=SimpleNamespace(id=1001))
        self.assertTrue(bot.authorized(message, users))
        message.chat.type = "group"
        self.assertFalse(bot.authorized(message, users))
        message.chat.type = "private"
        message.from_user.id = 1003
        self.assertFalse(bot.authorized(message, users))
        message.from_user = None
        self.assertFalse(bot.authorized(message, users))
        for value in ("", "-1", "not-a-number"):
            with self.assertRaises(ValueError):
                bot.allowed_user_ids(value)

    def test_untrusted_output_is_plain_text_and_chunked(self):
        client = SimpleNamespace(send_message=AsyncMock())
        text = "<b>untrusted</b>" + "😀" * 6500
        self.assertTrue(asyncio.run(bot.safe_send_message(client, 1001, text)))
        calls = client.send_message.call_args_list
        self.assertEqual("".join(c.args[1] for c in calls), text)
        self.assertTrue(all(c.kwargs["parse_mode"] is None and len(c.args[1].encode("utf-16-le")) // 2 <= 4096 for c in calls))

    def test_startup_does_not_connect_without_opt_in(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(bot, "load_dotenv"), \
                patch.object(bot, "Bot") as client:
            with self.assertRaises(ValueError):
                asyncio.run(bot.main())
            client.assert_not_called()

    def test_real_image_handler_uses_pipeline_and_plain_text_delivery(self):
        async def exercise():
            async def download(path, destination, timeout):
                destination.write(image_bytes("RGBA"))
            client = SimpleNamespace(
                get_file=AsyncMock(return_value=SimpleNamespace(file_size=100, file_path="demo.png")),
                download_file=AsyncMock(side_effect=download), send_message=AsyncMock())
            dispatcher = bot.build_dispatcher(client, {1001})
            handler = dispatcher.message.handlers[1].callback
            message = SimpleNamespace(chat=SimpleNamespace(type="private", id=1001),
                from_user=SimpleNamespace(id=1001), photo=[],
                document=SimpleNamespace(file_size=100, file_id="synthetic-image"))
            with patch.object(api, "_post", side_effect=[
                    {"textAnnotation": {"fullText": "fictional conversation"}},
                    {"alternatives": [{"message": {"text": "<b>suggestion</b>"}}]}]):
                await handler(message)
            client.send_message.assert_awaited_once_with(1001, "<b>suggestion</b>", parse_mode=None)
            client.download_file.reset_mock()
            message.from_user.id = 1002
            await handler(message)
            client.download_file.assert_not_called()
        asyncio.run(exercise())

    def test_handler_hides_private_provider_errors(self):
        async def exercise():
            client = SimpleNamespace(
                get_file=AsyncMock(side_effect=RuntimeError("private credentials")),
                send_message=AsyncMock())
            dispatcher = bot.build_dispatcher(client, {1001})
            message = SimpleNamespace(chat=SimpleNamespace(type="private", id=1001),
                from_user=SimpleNamespace(id=1001), photo=[],
                document=SimpleNamespace(file_size=100, file_id="synthetic-image"))
            await dispatcher.message.handlers[1].callback(message)
            self.assertNotIn("private credentials", str(client.send_message.call_args))
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
