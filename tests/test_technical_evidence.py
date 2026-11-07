import io

from PIL import Image, PngImagePlugin

from aidetect.technical_evidence import inspect_image


def png_bytes(metadata=None):
    image = Image.new("RGB", (32, 24), (120, 120, 120))
    output = io.BytesIO()
    png_info = None
    if metadata:
        png_info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            png_info.add_text(key, value)
    image.save(output, "PNG", pnginfo=png_info)
    return output.getvalue()


def test_technical_evidence_reports_verified_file_facts():
    result = inspect_image(png_bytes({"parameters": "steps=20"}))

    assert result["status"] == "inspected"
    assert result["pixel_facts"]["width"] == 32
    assert result["pixel_facts"]["height"] == 24
    assert result["pixel_facts"]["format"] == "PNG"
    assert isinstance(result["pixel_facts"]["estimated_entropy"], float)
    assert any("parameters" in item for item in result["embedded_signals"])


def test_technical_evidence_handles_invalid_bytes():
    result = inspect_image(b"not an image")

    assert result["status"] == "inspection_failed"
    assert result["limitations"]
