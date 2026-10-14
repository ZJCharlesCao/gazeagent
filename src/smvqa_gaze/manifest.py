from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(transfer_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(transfer_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        files.append(
            {
                "path": path.relative_to(transfer_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "files": files,
    }


def write_manifest(transfer_dir: Path) -> Path:
    manifest = build_manifest(transfer_dir)
    destination = transfer_dir / "MANIFEST.json"
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def verify_manifest(transfer_dir: Path) -> list[str]:
    manifest_path = transfer_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    expected = {row["path"]: row for row in manifest["files"]}
    observed = {
        path.relative_to(transfer_dir).as_posix()
        for path in transfer_dir.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    for relative, row in expected.items():
        path = transfer_dir / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        if path.stat().st_size != row["size_bytes"]:
            errors.append(f"size mismatch: {relative}")
            continue
        if sha256_file(path) != row["sha256"]:
            errors.append(f"checksum mismatch: {relative}")
    errors.extend(f"unexpected: {relative}" for relative in sorted(observed - set(expected)))
    return errors


def create_tar_archive(transfer_dir: Path, archive_path: Path) -> None:
    transfer_dir = transfer_dir.resolve()
    archive_path = archive_path.resolve()
    if archive_path == transfer_dir or transfer_dir in archive_path.parents:
        raise ValueError("The archive must be outside the transfer directory")
    # Archive exactly what was verified: never regenerate the manifest here.
    if not (transfer_dir / "MANIFEST.json").is_file():
        write_manifest(transfer_dir)
    errors = verify_manifest(transfer_dir)
    if errors:
        raise ValueError("Transfer directory does not match MANIFEST.json:\n" + "\n".join(errors[:20]))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w") as archive:
        archive.add(transfer_dir, arcname=transfer_dir.name)
