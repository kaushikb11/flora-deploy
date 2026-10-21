"""Test-only helpers for the media parity tests.

* ``load_nv_media``: imports NVIDIA's ``batch_utils/media.py`` and ``api.py`` from the local code
  cache under a throwaway package name, WITHOUT running ``batch_utils/__init__`` (which imports
  ``api`` -> fine) or ``bootstrap`` (which mutates ``sys.path``): ``media.py`` needs only ``av``
  and ``.config``; ``api.py`` needs ``.config`` and imports ``.media`` lazily. NVIDIA's
  ``canonical_rgb`` (``vendor/support/canonical_rgb.py``) is loaded by path; ``decode_selected``
  imports it as a top-level ``canonical_rgb`` module, so tests register it in ``sys.modules``.
  Nothing is copied: the code is read from ``~/.cache/flora-deploy/nv/...`` at test time. The
  real linker is WS-B's job (``flora_deploy.nv``).
* ``build_fixtures``: generates every fixture episode with ``flora_deploy.media.synth``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from flora_deploy.media import synth
from flora_deploy.media.synth import Spec
from flora_deploy.media.types import SourceRef

_ALIAS = "_flora_test_nv_tempseg_batch_utils"


@dataclass
class NvMedia:
    media: types.ModuleType
    api: types.ModuleType
    canonical_rgb: types.ModuleType


def load_nv_media(code_root: Path) -> NvMedia:
    bu = code_root / "batch_utils"
    if _ALIAS not in sys.modules:
        pkg = types.ModuleType(_ALIAS)
        pkg.__path__ = [str(bu)]  # package without executing its __init__.py
        sys.modules[_ALIAS] = pkg
    media = importlib.import_module(f"{_ALIAS}.media")
    api = importlib.import_module(f"{_ALIAS}.api")
    name = f"{_ALIAS}_canonical_rgb"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, bu / "vendor" / "support" / "canonical_rgb.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return NvMedia(media=media, api=api, canonical_rgb=sys.modules[name])


@dataclass
class Episode:
    name: str
    cameras: dict[str, SourceRef]  # role -> range in a (concatenated) file
    keyframe_aligned: bool = True
    three_view_canonical: bool = False  # 3-view and canonical-branch eligible frames
    notes: str = ""
    expect_probe_error: bool = False
    tags: set[str] = field(default_factory=set)


def _v3_file(out: Path, scratch: Path, specs: list[Spec]) -> list[SourceRef]:
    """Encode episodes separately, concatenate LeRobot-style, return each episode's range."""
    parts, refs, t = [], [], 0.0
    for k, s in enumerate(specs):
        p = synth.encode(scratch / f"{out.stem}_ep{k}.mp4", s)
        d = synth.video_duration_s(p)
        refs.append((t, t + d))
        t += d
        parts.append(p)
    synth.concat_like_lerobot(parts, out)
    return [SourceRef(str(out), a, b) for a, b in refs]


def build_fixtures(root: Path) -> list[Episode]:
    root.mkdir(parents=True, exist_ok=True)
    sc = root / "scratch"
    eps: list[Episode] = []

    # 1. H.264 CFR with B-frames, 3 episodes of different lengths in one v3-style file.
    refs = _v3_file(
        root / "h264_cfr.mp4", sc, [Spec(20, seed=1), Spec(37, seed=2), Spec(25, seed=3)]
    )
    eps += [
        Episode(f"h264_cfr_ep{i}", {"main": r}, tags={"h264", "cfr"}) for i, r in enumerate(refs)
    ]

    # 2. H.264 VFR (1/90000 time base, jittered PTS).
    def vfr(n, seed):
        return Spec(
            n,
            seed=seed,
            time_base=Fraction(1, 90000),
            pts=[i * 3000 + ((i * 7 + seed) % 5) * 330 for i in range(n)],
        )

    refs = _v3_file(root / "h264_vfr.mp4", sc, [vfr(24, 4), vfr(31, 5)])
    eps += [
        Episode(f"h264_vfr_ep{i}", {"main": r}, tags={"h264", "vfr"}) for i, r in enumerate(refs)
    ]

    # 3. AV1 (LeRobot's default codec): SVT-AV1 and libaom, decoded by libdav1d.
    refs = _v3_file(
        root / "av1_svt.mp4",
        sc,
        [
            Spec(18, codec="libsvtav1", size=(64, 64), seed=6),
            Spec(22, codec="libsvtav1", size=(64, 64), seed=7, gop=2),
        ],
    )
    eps += [Episode(f"av1_svt_ep{i}", {"main": r}, tags={"av1"}) for i, r in enumerate(refs)]
    refs = _v3_file(
        root / "av1_aom.mp4",
        sc,
        [Spec(16, codec="libaom-av1", seed=8), Spec(19, codec="libaom-av1", seed=9)],
    )
    eps += [Episode(f"av1_aom_ep{i}", {"main": r}, tags={"av1"}) for i, r in enumerate(refs)]

    # 4. Mixed rates in one file (same 1/15360 time base): whole-file average_rate is neither.
    # No B-frames: different reorder delays would make the concat demuxer's dts non-monotonic.
    tb = Fraction(1, 15360)
    refs = _v3_file(
        root / "mixed_rate.mp4",
        sc,
        [
            Spec(30, seed=10, time_base=tb, pts=[i * 512 for i in range(30)], bframes=0),
            Spec(15, rate=15, seed=11, time_base=tb, pts=[i * 1024 for i in range(15)], bframes=0),
            Spec(
                20,
                rate=25,
                seed=12,
                time_base=tb,
                pts=[round(i * 614.4) for i in range(20)],
                bframes=0,
            ),
        ],
    )
    eps += [Episode(f"mixed_rate_ep{i}", {"main": r}, tags={"mixed"}) for i, r in enumerate(refs)]

    # 5. One long encode split mid-GOP: only episode 0 starts on a keyframe.
    long = synth.encode(root / "one_gop.mp4", Spec(60, seed=13, gop=1000, bframes=0))
    for i, (a, b) in enumerate([(0, 20), (20, 40), (40, 60)]):
        eps.append(
            Episode(
                f"one_gop_ep{i}",
                {"main": SourceRef(str(long), a / 30, b / 30)},
                keyframe_aligned=(i == 0),
                tags={"gop"},
            )
        )

    # 6. Non-standard last-sample durations (stts tail) - whole files and inside a v3 file.
    base = synth.encode(sc / "tail_base.mp4", Spec(120, seed=14, bframes=0))
    t20 = synth.remux_with_last_duration(base, root / "tail_x20.mp4", 512 * 20)
    base2 = synth.encode(sc / "tail_base2.mp4", Spec(40, seed=15))
    t3 = synth.remux_with_last_duration(base2, root / "tail_x3.mp4", 512 * 3)
    eps += [
        Episode("tail_x20_whole", {"main": SourceRef(str(t20))}, tags={"tail"}),
        Episode("tail_x3_whole", {"main": SourceRef(str(t3))}, tags={"tail"}),
    ]
    parts = [t3, synth.encode(sc / "tail_next.mp4", Spec(20, seed=16))]
    d = [synth.video_duration_s(p) for p in parts]
    synth.concat_like_lerobot(parts, root / "tail_cat.mp4")
    eps += [
        Episode(
            "tail_cat_ep0",
            {"main": SourceRef(str(root / "tail_cat.mp4"), 0.0, d[0])},
            tags={"tail"},
        ),
        Episode(
            "tail_cat_ep1",
            {"main": SourceRef(str(root / "tail_cat.mp4"), d[0], d[0] + d[1])},
            tags={"tail"},
        ),
    ]

    # 7. Three synchronized cameras (concatenated per camera), canonical-eligible and BT.709.
    for label, kw in (("canon", {}), ("bt709", {"colorspace": 1, "color_range": 1})):
        cams = {}
        for c, role in enumerate(("main", "left_wrist", "right_wrist")):
            cams[role] = _v3_file(
                root / f"three_{label}_{role}.mp4",
                sc,
                [Spec(21, seed=20 + c, **kw), Spec(26, seed=30 + c, **kw)],
            )
        for i in range(2):
            eps.append(
                Episode(
                    f"three_{label}_ep{i}",
                    {r: cams[r][i] for r in cams},
                    three_view_canonical=(label == "canon"),
                    tags={"three"},
                )
            )

    # 8. Single-view colour variants: full range (yuvj420p), width not a multiple of 16.
    for label, s in (
        ("fullrange", Spec(12, seed=40, color_range=2)),
        ("w72", Spec(12, seed=41, size=(72, 48))),
    ):
        p = synth.encode(root / f"{label}.mp4", s)
        eps.append(Episode(label, {"main": SourceRef(str(p))}, tags={"color"}))
    return eps


PROBE_FIELDS = ("rate", "pts_us", "duration_us", "size", "cfr")


@dataclass
class Adversarial:
    """A source the index must either match NVIDIA on exactly, or refuse with ``reasons``."""

    name: str
    ref: SourceRef
    reasons: frozenset[str] = frozenset()  # expected ineligibility reasons (empty = eligible)
    frames: int | None = None  # frames the source decoder outputs for the range
    # Whole-file sources: fields that must also equal NVIDIA probe(original file).
    original_fields: tuple[str, ...] = ()
    notes: str = ""


def build_adversarial(root: Path) -> list[Adversarial]:
    """Real-world shapes the clean fixtures miss (review of WS-C: H1, H2, M1, M2, L2, L3)."""
    root.mkdir(parents=True, exist_ok=True)
    sc = root / "scratch"
    out: list[Adversarial] = []

    # H1a: `ffmpeg -ss 1.5 -c copy` style trim - a 15-packet pre-roll flagged DISCARD.
    base = synth.encode(sc / "trim_src.mp4", Spec(90, seed=50, gop=30))
    trim = synth.remux_trimmed(base, root / "edit_trim.mp4", from_keyframe=1, trim_ticks=45 * 512)
    out.append(
        Adversarial(
            "edit_trim",
            SourceRef(str(trim)),
            frozenset({"edit_list"}),
            frames=45,
            original_fields=PROBE_FIELDS,
        )
    )
    # H1b: zero-length last sample -> movenc ends the edit list before it -> last packet DISCARD.
    # The original's avg_frame_rate counts the hidden sample's zero stts, so only PTS/duration
    # are compared with probe(original).
    b0 = synth.encode(sc / "last0_src.mp4", Spec(40, seed=51, bframes=0))
    last0 = synth.remux_with_last_duration(b0, root / "edit_last0.mp4", 0)
    out.append(
        Adversarial(
            "edit_last0",
            SourceRef(str(last0)),
            frozenset({"edit_list"}),
            frames=39,
            original_fields=("pts_us", "duration_us", "size"),
        )
    )

    # H2: x264 periodic intra refresh: sync samples every 30 frames, only frame 0 is an IDR.
    ir = synth.encode(
        root / "intra_refresh.mp4",
        Spec(90, seed=52, bframes=0, x264_params="intra-refresh=1:keyint=30:scenecut=0"),
    )
    out.append(Adversarial("intra_refresh_ep0", SourceRef(str(ir), 0.0, 1.0), frames=30))
    for k in (1, 2):
        out.append(
            Adversarial(
                f"intra_refresh_ep{k}",
                SourceRef(str(ir), float(k), k + 1.0),
                frozenset({"not_random_access"}),
                frames=30,
            )
        )
    # M1: whole file that starts 10 packets into a GOP (segment split); h264 drops frames
    # until the next IDR, so the file decodes to 60 frames, not 80 packets.
    g = synth.encode(sc / "nokf_src.mp4", Spec(90, seed=53, gop=30, bframes=0))
    nokf = synth.remux_drop_head(g, root / "nokf_whole.mp4", 10)
    out.append(
        Adversarial(
            "nokf_whole",
            SourceRef(str(nokf)),
            frozenset({"not_keyframe"}),
            frames=60,
            original_fields=PROBE_FIELDS,
        )
    )
    # M2: B-frame GOP split mid-pyramid: the range is not contiguous in decode order.
    nc = synth.encode(root / "noncontig.mp4", Spec(90, seed=1, gop=30))
    out.append(
        Adversarial(
            "noncontig",
            SourceRef(str(nc), 1.0, 50 / 30),
            frozenset({"non_contiguous"}),
            frames=20,
        )
    )
    # L3: .mov with an nclc colr box (no range field): frame color_range 0 in the source, 1 in
    # the MP4 cut. Canonical-eligible both ways (BT.470BG), so pixels must not change.
    mov = synth.encode(root / "colr.mov", Spec(40, seed=54, colorspace=5))
    out.append(
        Adversarial("mov_nclc", SourceRef(str(mov)), frames=40, original_fields=PROBE_FIELDS)
    )
    # L2: track timescale 30 (< 10000): the cut gets 1/15360.
    ts = synth.encode(
        root / "ts30.mp4",
        Spec(45, seed=55, bframes=0, gop=15, time_base=Fraction(1, 30), track_timescale=30),
    )
    out.append(Adversarial("timescale_30", SourceRef(str(ts)), frames=45))
    out.append(Adversarial("timescale_30_ep1", SourceRef(str(ts), 0.5, 1.5), frozenset(), 30))
    # HEVC (bitstream RAP check: IRAP NAL types), closed GOP every 20 frames.
    hevc = synth.encode(
        root / "hevc.mp4",
        Spec(
            60,
            seed=56,
            codec="libx265",
            codec_options={
                "x265-params": "keyint=20:min-keyint=20:scenecut=0:bframes=0:log-level=error"
            },
        ),
    )
    for k in range(3):
        out.append(
            Adversarial(
                f"hevc_ep{k}", SourceRef(str(hevc), k * 20 / 30, (k + 1) * 20 / 30), frames=20
            )
        )
    return out


MANY_LENGTHS = tuple(3 + (k * 5) % 9 for k in range(24))


def build_many_episodes(root: Path) -> list[Adversarial]:
    """A LeRobot-style v3 file: SVT-AV1 with g=2 (LeRobot's settings), many short episodes
    encoded separately and concatenated."""
    specs = [
        Spec(n, codec="libsvtav1", size=(64, 64), seed=60 + k, gop=2)
        for k, n in enumerate(MANY_LENGTHS)
    ]
    refs = _v3_file(root / "many.mp4", root / "scratch_many", specs)
    return [
        Adversarial(f"many_ep{k}", r, frames=n)
        for k, (r, n) in enumerate(zip(refs, MANY_LENGTHS, strict=True))
    ]


def build_envelope_fixtures(root: Path) -> dict[str, dict[str, SourceRef]]:
    """Cameras -> refs for files NVIDIA must reject (or accept) at the envelope boundary."""
    root.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, SourceRef]] = {}

    def one(name, spec):
        out[name] = {"main": SourceRef(str(synth.encode(root / f"{name}.mp4", spec)))}

    one("fps_3", Spec(10, rate=3, size=(16, 16)))
    one("fps_4", Spec(10, rate=4, size=(16, 16)))
    one("one_frame", Spec(1, size=(16, 16)))
    one("dur_121s", Spec(605, rate=5, size=(16, 16), bframes=0))
    one("dur_120s", Spec(600, rate=5, size=(16, 16), bframes=0))
    one("frames_7201", Spec(7201, rate=100, size=(16, 16), bframes=0))
    one("frames_7200", Spec(7200, rate=100, size=(16, 16), bframes=0))
    one("wide_4112", Spec(3, size=(4112, 16)))
    # 3-view: a camera with an extra frame, and VFR cameras with identical timestamps.
    sync_bad = {}
    for c, role in enumerate(("main", "left_wrist", "right_wrist")):
        sync_bad[role] = SourceRef(
            str(synth.encode(root / f"desync_{role}.mp4", Spec(20 + (c == 2), seed=c)))
        )
    out["three_desync"] = sync_bad
    vfr = {}
    for c, role in enumerate(("main", "left_wrist", "right_wrist")):
        vfr[role] = SourceRef(
            str(
                synth.encode(
                    root / f"vfr3_{role}.mp4",
                    Spec(
                        20,
                        seed=c,
                        time_base=Fraction(1, 90000),
                        pts=[i * 3000 + (i % 3) * 700 for i in range(20)],
                    ),
                )
            )
        )
    out["three_vfr"] = vfr
    return out
