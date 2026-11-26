"""Extract small TotalSegmentator L5 masks from a verified HTTP ZIP range source."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
from zipfile import ZipFile

import requests


SOURCE = "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip"
SIZE = 23581218285
MEMBER = re.compile(r"(?:^|/)(s[0-9]+)/segmentations/vertebrae_L5\.nii\.gz$")
TRANSFER_SUBJECT = "s0010"


class RemoteArchive(io.RawIOBase):
    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.position = 0
        self.session = requests.Session()
        self.bytes = 0
        self.requests = 0
        self.cache = (0, b'')

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Unknown seek origin")
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise ValueError("Invalid seek position")
        return self.position

    def read(self, n=-1):
        left = self.position
        right = self.size - 1 if n < 0 else min(self.size - 1, left + n - 1)
        if right < left:
            return b""
        length = right - left + 1
        start, chunk = self.cache
        if start <= left and right < start + len(chunk):
            self.position += length
            return chunk[left - start:right + 1 - start]
        if length < 4096:
            right = min(self.size - 1, max(right, left + 65535))
        expected_length = right - left + 1
        if length > 32 * 1024 * 1024:
            raise ValueError(f"Refused archive read of {length} bytes")
        expect = f"bytes {left}-{right}/{self.size}"
        headers = {"Range": f"bytes={left}-{right}", "Accept-Encoding": "identity"}
        for trial in range(3):
            # Some intermediaries cache a response without accounting for Range.
            url = self.url if trial == 0 else f"{self.url}?range_start={left}&range_end={right}"
            try:
                with self.session.get(url, headers=headers, timeout=(15, 40), stream=True) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != expect:
                        raise ValueError(f"Expected HTTP 206 {expect}, got {response.status_code} "
                                         f"{response.headers.get('Content-Range')}")
                    body = response.raw.read(expected_length + 1)
                if len(body) != expected_length:
                    raise ValueError(f"Incorrect range length {len(body)} for {expect}")
            except (requests.RequestException, OSError, ValueError) as exc:
                if trial == 2:
                    raise RuntimeError("The ZIP range request failed") from exc
                time.sleep(1 + trial)
            else:
                self.requests += 1
                self.bytes += expected_length
                self.position += length
                self.cache = (left, body)
                return body[:length]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selection_key(name, salt):
    return sha((salt + "/" + name).encode())


def run(output, count, salt, exclude_path, source=SOURCE, size=SIZE, prior=None, body_coverage=False):
    output.mkdir(parents=True, exist_ok=True)
    masks = output / "masks"
    masks.mkdir(exist_ok=True)
    exclude = set(exclude_path.read_text().split()) | {TRANSFER_SUBJECT}
    remote = RemoteArchive(source, size)
    begin = time.perf_counter()
    with ZipFile(remote) as archive:
        entries = {}
        for info in archive.infolist():
            match = MEMBER.search(info.filename)
            if match:
                subject = match[1]
                if subject in entries:
                    raise ValueError(f"Repeated L5 candidate {subject}")
                entries[subject] = info
        metadata = [i for i in archive.namelist() if i.split("/")[-1] == "meta.csv"]
        if len(metadata) != 1 or TRANSFER_SUBJECT not in entries:
            raise ValueError(f"Unexpected official archive index, {len(entries)} masks and {metadata}")
        payload = archive.read(metadata[0])
        (output / "original_meta.csv").write_bytes(payload)
        table = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")), delimiter=";")
        metadata_rows = list(table)
        header = table.fieldnames
        if len({row["image_id"] for row in metadata_rows}) != len(metadata_rows):
            raise ValueError("Repeated source image IDs")
        initial = [] if prior is None else json.loads(prior.read_text())["entries"]
        allowed = entries.keys() - exclude - {row["subject_id"] for row in initial}
        if body_coverage:
            body_types = {"ct abdomen-pelvis", "ct neck-thorax-abdomen-pelvis",
                          "ct thorax-abdomen-pelvis", "ct pelvis", "ct polytrauma", "ct whole body"}
            allowed &= {row["image_id"] for row in metadata_rows if row["study_type"].strip().lower() in body_types}
        eligible = sorted(allowed, key=lambda p: selection_key(p, salt))
        print(f"Official ZIP has {len(entries)} L5 mask entries. {len(eligible)} remain after "
              f"excluding local IDs. Metadata columns: {header}", flush=True)
        manifest = {"source": source, "record": "10.5281/zenodo.10047292", "license": "CC BY 4.0",
                    "metadata_entry": metadata[0], "metadata_sha256": sha(payload),
                    "transfer_id_present_in_index": True, "excluded_ids": sorted(exclude),
                    "indexed_masks": len(entries), "eligible_indexed_masks": len(eligible),
                    "selection_salt": salt, "entries": initial.copy(),
                    "supplemental_study_type_filter": body_coverage,
                    "initial_manifest_sha256": sha(prior.read_bytes()) if prior else None}
        for position, subject in enumerate(eligible[:count], start=1):
            info = entries[subject]
            if info.file_size > 4 * 1024 * 1024:
                raise ValueError("Unexpected L5 member size")
            path = masks / f"{subject}.nii.gz"
            body = archive.read(info)
            if path.exists() and sha(path.read_bytes()) != sha(body):
                raise ValueError(f"Existing L5 member differs for {subject}")
            path.write_bytes(body)
            manifest["entries"].append({"subject_id": subject, "source_member": info.filename,
                                        "mask": str(path.resolve()), "bytes": len(body),
                                        "crc32": f"{info.CRC:08x}", "sha256": sha(body)})
            print(f"{position}/{count} new masks ({remote.bytes / 1e6:.1f} MB received)",
                  flush=True)
            manifest["bytes_received"] = remote.bytes
            manifest["range_requests"] = remote.requests
            manifest["seconds"] = time.perf_counter() - begin
            (output / "acquisition.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=56)
    parser.add_argument("--exclude", required=True, type=Path)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--initial-manifest", type=Path)
    parser.add_argument("--body-coverage", action="store_true")
    a = parser.parse_args()
    run(a.output, a.count, a.salt, a.exclude, a.source, a.size, a.initial_manifest, a.body_coverage)
