"""Guards on the frozen-dataset checks that gate a baseline run.

These decide whether a training run is allowed to start, so a bug here is worse
than a bug in the run itself: it would let a run train on the wrong data and
still look like evidence. No GPU, no Ultralytics — the export is a handful of
files on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpr.baseline import (
    BaselineError,
    baseline_provenance,
    export_fingerprint,
    find_macos_metadata,
    inspect_dataset,
    is_macos_metadata,
    label_files,
    sha256_file,
    verify_dataset,
    write_data_yaml,
)


def make_export(root, counts=(2, 1, 1), boxes_per_image=1):
    """A minimal YOLO export: images/<split>/ and labels/<split>/."""
    for split, n in zip(("train", "val", "test"), counts, strict=True):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (root / "images" / split / f"{split}_{i}.jpg").write_bytes(b"\xff\xd8fake")
            (root / "labels" / split / f"{split}_{i}.txt").write_text(
                "0 0.5 0.5 0.2 0.1\n" * boxes_per_image
            )
    (root / "data.yaml").write_text("path: /somewhere/else\ntrain: images/train\n")
    return root


def make_manifest(path, records=4):
    path.write_text("".join(json.dumps({"image_id": f"i{i}"}) + "\n" for i in range(records)))
    return path


class TestExportFingerprint:
    def test_is_stable_for_the_same_content(self, tmp_path):
        a = make_export(tmp_path / "a")
        b = make_export(tmp_path / "b")
        assert export_fingerprint(a) == export_fingerprint(b)

    def test_is_independent_of_location_and_of_data_yaml(self, tmp_path):
        # data.yaml embeds an absolute path, so including it would make the
        # fingerprint useless after a transfer — the one thing it must survive.
        a = make_export(tmp_path / "a")
        b = make_export(tmp_path / "b")
        (b / "data.yaml").write_text("path: /a/completely/different/place\n")
        assert export_fingerprint(a) == export_fingerprint(b)

    def test_changes_when_an_annotation_changes(self, tmp_path):
        a = make_export(tmp_path / "a")
        before = export_fingerprint(a)
        (a / "labels" / "train" / "train_0.txt").write_text("0 0.4 0.4 0.2 0.1\n")
        assert export_fingerprint(a) != before

    def test_changes_when_an_image_moves_split(self, tmp_path):
        a = make_export(tmp_path / "a")
        before = export_fingerprint(a)
        src = a / "labels" / "train" / "train_0.txt"
        src.rename(a / "labels" / "test" / "train_0.txt")
        assert export_fingerprint(a) != before, "a split change must change the fingerprint"


class TestInspectDataset:
    def test_counts_images_labels_and_annotations(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(3, 2, 1), boxes_per_image=2)
        facts = inspect_dataset(root)
        assert facts.labels == {"train": 3, "val": 2, "test": 1}
        assert facts.images == {"train": 3, "val": 2, "test": 1}
        assert facts.annotations == 12
        assert facts.total_images == 6

    def test_reads_manifest_when_given(self, tmp_path):
        root = make_export(tmp_path / "e")
        manifest = make_manifest(tmp_path / "manifest.jsonl", records=4)
        facts = inspect_dataset(root, manifest)
        assert facts.manifest_records == 4
        assert facts.manifest_sha256 == sha256_file(manifest)

    def test_counts_symlinked_images(self, tmp_path):
        # The canonical export is 3,105 symlinks; a dereferenced copy is regular
        # files. Both must measure identically or the transfer check is useless.
        root = make_export(tmp_path / "e", counts=(2, 1, 1))
        real = tmp_path / "real.jpg"
        real.write_bytes(b"\xff\xd8fake")
        link_dir = root / "images" / "train"
        for p in list(link_dir.iterdir()):
            p.unlink()
            p.symlink_to(real)
        assert inspect_dataset(root).images["train"] == 2

    def test_missing_label_directory_is_an_error(self, tmp_path):
        root = make_export(tmp_path / "e")
        for f in (root / "labels" / "test").iterdir():
            f.unlink()
        (root / "labels" / "test").rmdir()
        with pytest.raises(BaselineError, match="missing label directory"):
            inspect_dataset(root)

    def test_missing_manifest_is_an_error(self, tmp_path):
        root = make_export(tmp_path / "e")
        with pytest.raises(BaselineError, match="manifest not found"):
            inspect_dataset(root, tmp_path / "nope.jsonl")


def add_appledouble(root):
    """Drop a sidecar beside every real file, as a Linux extraction would.

    The bytes are the real AppleDouble magic — a binary attribute blob, not
    text — so anything that opens one as a label gets what it deserves.
    """
    blob = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \x00\x02\x00\x00\x00\tATTR"
    made = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("._"):
            sidecar = path.with_name(f"._{path.name}")
            sidecar.write_bytes(blob)
            made.append(sidecar)
    return made


class TestAppleDoubleFiles:
    """macOS sidecars must never be mistaken for dataset content.

    `._label.txt` matches `*.txt`, so a naive glob counts one label twice and
    doubles the annotation total. These files appear on the *destination*:
    archiving on macOS carries extended attributes along, and extracting on
    Linux materialises them as real files — so a pristine dataset can still
    arrive contaminated.
    """

    def test_a_sidecar_is_not_a_label(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(2, 1, 1))
        (root / "labels" / "train" / "._example.txt").write_bytes(b"Mac OS X\x00ATTR")
        facts = inspect_dataset(root)
        assert facts.labels["train"] == 2, "an AppleDouble file was counted as a label"

    def test_annotations_are_not_inflated(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(2, 1, 1), boxes_per_image=1)
        add_appledouble(root)
        assert inspect_dataset(root).annotations == 4

    def test_a_sidecar_is_not_an_image(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(2, 1, 1))
        (root / "images" / "train" / "._example.jpg").write_bytes(b"Mac OS X\x00ATTR")
        assert inspect_dataset(root).images["train"] == 2

    def test_the_fingerprint_ignores_them(self, tmp_path):
        # Identity must survive a contaminated transfer, or the hash could never
        # tell "wrong dataset" apart from "same dataset, bad copy".
        root = make_export(tmp_path / "e")
        before = export_fingerprint(root)
        add_appledouble(root)
        assert export_fingerprint(root) == before

    def test_they_are_found_and_listed(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(1, 1, 1))
        made = add_appledouble(root)
        found = find_macos_metadata(root)
        assert len(found) == len(made)
        assert all(Path(f).name.startswith("._") for f in found)

    def test_verify_rejects_them_even_when_everything_else_matches(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(2, 1, 1))
        clean = inspect_dataset(root)
        add_appledouble(root)
        dirty = inspect_dataset(root)
        # Counts and hash are still correct — only the sidecars differ.
        assert dirty.export_sha256 == clean.export_sha256
        assert dirty.labels == clean.labels
        with pytest.raises(BaselineError, match="macOS metadata file"):
            verify_dataset(
                dirty,
                export_sha256=clean.export_sha256,
                counts={"train": 2, "val": 1, "test": 1},
                annotations=4,
            )

    def test_the_error_says_how_to_fix_it(self, tmp_path):
        root = make_export(tmp_path / "e")
        add_appledouble(root)
        with pytest.raises(BaselineError) as exc:
            verify_dataset(inspect_dataset(root), export_sha256="0" * 64)
        assert "COPYFILE_DISABLE=1" in str(exc.value)

    @pytest.mark.parametrize(
        ("name", "is_metadata"),
        [
            ("._example.txt", True),
            ("._eu-test-plate_jpg.rf.abc123.txt", True),
            (".DS_Store", True),
            ("__MACOSX", True),
            ("example.txt", False),
            ("eu-test-plate_jpg.rf.abc123.txt", False),
            # A real label may legitimately contain "._" without starting with it.
            ("clip._frame.txt", False),
        ],
    )
    def test_classification(self, name, is_metadata):
        assert is_macos_metadata(name) is is_metadata

    def test_label_files_is_the_only_safe_enumerator(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(2, 1, 1))
        (root / "labels" / "train" / "._example.txt").write_bytes(b"Mac OS X")
        raw = sorted((root / "labels" / "train").glob("*.txt"))
        safe = label_files(root / "labels" / "train")
        assert len(raw) == 3, "fixture should have produced a sidecar"
        assert len(safe) == 2
        assert all(not p.name.startswith("._") for p in safe)

    def test_a_genuinely_undecodable_label_still_raises(self, tmp_path):
        # Guards against 'fixing' this with errors="ignore" or latin-1: a real
        # label full of binary must fail loudly, not decode to nonsense and
        # quietly shrink the dataset.
        root = make_export(tmp_path / "e", counts=(1, 1, 1))
        (root / "labels" / "train" / "train_0.txt").write_bytes(b"\xff\xfe\x00\x01binary")
        with pytest.raises(UnicodeDecodeError):
            inspect_dataset(root)

    def test_a_clean_export_reports_no_metadata(self, tmp_path):
        assert inspect_dataset(make_export(tmp_path / "e")).macos_metadata == []


class TestVerifyDataset:
    def _facts(self, tmp_path):
        root = make_export(tmp_path / "e", counts=(2, 1, 1))
        manifest = make_manifest(tmp_path / "m.jsonl", records=4)
        return inspect_dataset(root, manifest)

    def test_passes_when_everything_matches(self, tmp_path):
        facts = self._facts(tmp_path)
        verify_dataset(
            facts,
            export_sha256=facts.export_sha256,
            manifest_sha256=facts.manifest_sha256,
            counts={"train": 2, "val": 1, "test": 1},
            annotations=4,
            manifest_records=4,
        )

    def test_rejects_a_wrong_export_hash(self, tmp_path):
        facts = self._facts(tmp_path)
        with pytest.raises(BaselineError, match="export sha256"):
            verify_dataset(facts, export_sha256="0" * 64)

    def test_rejects_a_wrong_manifest_hash(self, tmp_path):
        facts = self._facts(tmp_path)
        with pytest.raises(BaselineError, match="manifest sha256"):
            verify_dataset(facts, export_sha256=facts.export_sha256, manifest_sha256="0" * 64)

    def test_rejects_wrong_counts(self, tmp_path):
        facts = self._facts(tmp_path)
        with pytest.raises(BaselineError, match="train: 2 labels, expected 2175"):
            verify_dataset(facts, export_sha256=facts.export_sha256, counts={"train": 2175})

    def test_rejects_wrong_annotation_total(self, tmp_path):
        facts = self._facts(tmp_path)
        with pytest.raises(BaselineError, match="annotations 4, expected 3273"):
            verify_dataset(facts, export_sha256=facts.export_sha256, annotations=3273)

    def test_reports_every_problem_at_once(self, tmp_path):
        facts = self._facts(tmp_path)
        with pytest.raises(BaselineError) as exc:
            verify_dataset(
                facts,
                export_sha256="0" * 64,
                manifest_sha256="1" * 64,
                counts={"train": 999},
                annotations=3273,
            )
        message = str(exc.value)
        assert "export sha256" in message
        assert "manifest sha256" in message
        assert "train:" in message
        assert "annotations" in message


class TestWriteDataYaml:
    def test_points_at_the_current_location(self, tmp_path):
        root = make_export(tmp_path / "e")
        out = write_data_yaml(root, tmp_path / "e" / "data_colab.yaml")
        text = out.read_text()
        assert f"path: {root.resolve()}" in text
        assert "train: images/train" in text
        assert "0: license_plate" in text

    def test_refuses_to_overwrite_the_exports_own_data_yaml(self, tmp_path):
        # The canonical one is frozen read-only; this is the guard that stops a
        # careless path argument from trying.
        root = make_export(tmp_path / "e")
        original = (root / "data.yaml").read_text()
        with pytest.raises(BaselineError, match="refusing to overwrite"):
            write_data_yaml(root, root / "data.yaml")
        assert (root / "data.yaml").read_text() == original

    def test_leaves_the_original_untouched(self, tmp_path):
        root = make_export(tmp_path / "e")
        before = (root / "data.yaml").read_text()
        write_data_yaml(root, root / "data_colab.yaml")
        assert (root / "data.yaml").read_text() == before


class TestBaselineProvenance:
    def _record(self, tmp_path, **kwargs):
        facts = inspect_dataset(make_export(tmp_path / "e"))
        return baseline_provenance(
            dataset=facts,
            dataset_paths={"export": "/content/canonical/yolo"},
            train_config={"model": "yolov8s.pt", "epochs": 100, "seed": 0},
            resolved_optimizer=None,
            environment={"ultralytics": "8.4.117"},
            code={"git_commit": "abc123"},
            **kwargs,
        )

    def test_unmeasured_sections_stay_null(self, tmp_path):
        payload = self._record(tmp_path).payload
        assert payload["metrics"]["validation"] is None
        assert payload["metrics"]["test"] is None
        assert payload["resolved_optimizer"] is None
        assert payload["checkpoint"] is None

    def test_carries_the_dataset_identity(self, tmp_path):
        facts = inspect_dataset(make_export(tmp_path / "e"))
        payload = self._record(tmp_path).payload
        assert payload["dataset"]["export_sha256"] == facts.export_sha256
        assert payload["dataset"]["annotations"] == facts.annotations

    def test_states_the_single_intentional_change(self, tmp_path):
        payload = self._record(tmp_path).payload
        assert payload["intentional_change_vs_historical"] == (
            "historical dataset -> canonical dataset"
        )

    def test_round_trips_as_json(self, tmp_path):
        record = self._record(tmp_path, test_metrics={"mAP50": 0.5})
        out = record.write(tmp_path / "provenance.json")
        assert json.loads(out.read_text())["metrics"]["test"] == {"mAP50": 0.5}


class TestSha256File:
    def test_matches_hashlib(self, tmp_path):
        import hashlib

        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 5_000_000)
        assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
