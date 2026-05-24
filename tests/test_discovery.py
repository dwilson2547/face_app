from pathlib import Path

from face_poc.discovery import discover_images


def test_discover_images_filters_supported_suffixes(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "d.webp").write_bytes(b"x")

    recursive = discover_images(tmp_path)
    non_recursive = discover_images(tmp_path, recursive=False)

    assert [path.name for path in recursive] == ["a.jpg", "b.png", "d.webp"]
    assert [path.name for path in non_recursive] == ["a.jpg", "b.png"]
