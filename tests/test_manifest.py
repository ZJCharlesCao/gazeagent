import json

from smvqa_gaze.manifest import verify_manifest, write_manifest


def test_manifest_detects_changed_files(tmp_path):
    transfer = tmp_path / "transfer"
    transfer.mkdir()
    artifact = transfer / "qa.jsonl"
    artifact.write_text('{"id":1}\n', encoding="utf-8")
    write_manifest(transfer)
    assert verify_manifest(transfer) == []
    artifact.write_text('{"id":2}\n', encoding="utf-8")
    assert verify_manifest(transfer) == ["checksum mismatch: qa.jsonl"]
