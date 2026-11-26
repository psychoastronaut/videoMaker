"""CLI entry point for the Psycho Astronaut video pipeline."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.common import (
    ENV_CONFIG,
    authoring_signature,
    episode_dir,
    episodes_root,
    load_config,
    load_dotenv,
    read_json,
    resolved_signature,
    voice_path,
    write_json,
)

app = typer.Typer(
    name="videomaker",
    help="Psycho Astronaut — Video Automation Pipeline",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()

StageFunc = Callable[[str, Path], Any]
STATE_FILE = ".stage_state.json"


@app.callback()
def _main(
    config: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help="Alternative YAML config (default: config/default.yaml)"),
) -> None:
    load_dotenv()
    if config is not None:
        if not config.is_file():
            console.print(f"[red]Config not found: {config}[/]")
            raise typer.Exit(1)
        os.environ[ENV_CONFIG] = str(config.resolve())


def get_episode_dir(episode_id: str) -> Path:
    base = episode_dir(episode_id)
    for sub in ("raw", "work", "out"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    for sub in ("clips", "images", "schematics"):
        (base / "raw" / sub).mkdir(parents=True, exist_ok=True)
    return base


# --------------------------------------------------------------------------- stage table


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    module: str


STAGES = [
    Stage("audio", "Stage 0: Voice Chain", "src.audio_clean"),
    Stage("transcribe", "Stage 1: Transcribe", "src.stage1_transcribe"),
    Stage("index", "Stage 2: Index Assets", "src.stage2_index_assets"),
    Stage("semantic", "Stage 2a: Semantic Index", "src.semantic_index"),
    Stage("prefetch", "Stage 2b: Prefetch Stock Pool", "src.stage_prefetch"),
    Stage("plan", "Stage 3: Plan Scenes", "src.stage3_plan_scenes"),
    Stage("fetch", "Stage 4: Fetch Stock Media", "src.stage4_fetch_stock"),
    Stage("cutouts", "Stage 5: Cutouts & Overlays", "src.stage5_cutouts"),
    Stage("render", "Stage 6: Render Final Cut", "src.stage6_render"),
]
STAGE_BY_KEY = {s.key: s for s in STAGES}


def _stage_func(stage: Stage) -> StageFunc:
    import importlib

    return importlib.import_module(stage.module).run


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _newest(paths: list[Path]) -> float:
    return max((_mtime(p) for p in paths), default=0.0)


def _raw_files(ep: Path) -> list[Path]:
    files: list[Path] = []
    for sub in ("clips", "images", "schematics"):
        d = ep / "raw" / sub
        if d.is_dir():
            files += [p for p in d.iterdir() if p.is_file()]
    return files


def _load_state(work: Path) -> dict[str, str]:
    p = work / STATE_FILE
    try:
        return read_json(p) if p.is_file() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(work: Path, state: dict[str, str]) -> None:
    write_json(work / STATE_FILE, state)


def _plan_signatures(work: Path) -> tuple[str, str]:
    plan_path = work / "scene_plan.json"
    if not plan_path.is_file():
        return "", ""
    plan = read_json(plan_path)
    return authoring_signature(plan), resolved_signature(plan)


def stale_reason(key: str, ep: Path, work: Path) -> str | None:
    """Why a stage must (re)run, or None if its artifact is up to date.

    Stages 1-3 compare file mtimes with their inputs. Stages 4-5 mutate scene_plan.json
    themselves, so they compare content signatures of the plan recorded when they last ran.
    """
    state = _load_state(work)
    voice = voice_path(ep)
    if key == "audio":
        from src.audio_clean import find_source, is_enabled

        src = find_source(ep / "raw")
        if src is None or not is_enabled(ep.name) or _transcript_is_synthetic(work):
            return None  # (a synthetic demo transcript is timed to the raw audio)
        art = work / "voiceover_clean.wav"
        if not art.exists():
            return "no clean voice-over"
        return "voiceover newer than clean file" if _mtime(src) > _mtime(art) else None
    if key == "transcribe":
        art = work / "transcript.json"
        if not art.exists():
            return "no transcript"
        return "voiceover newer than transcript" if _mtime(voice) > _mtime(art) else None
    if key == "index":
        art = work / "local_assets.json"
        if not art.exists():
            return "no asset index"
        return "raw assets changed" if _newest(_raw_files(ep)) > _mtime(art) else None
    if key == "semantic":
        from src.semantic import get_backend

        art = work / "semantic_index.json"
        if not (work / "local_assets.json").exists():
            return "no asset index"
        backend = get_backend(quiet=True)
        if backend is None:
            return None  # nothing to build; planner falls back to keyword matching
        if not art.exists():
            return "no semantic index"
        if _mtime(work / "local_assets.json") > _mtime(art):
            return "asset index newer than semantic index"
        try:
            if read_json(art).get("tag") != backend.tag:
                return "semantic backend/model changed"
        except json.JSONDecodeError:
            return "semantic index is corrupt"
        return None
    if key == "prefetch":
        from src.stage_prefetch import prefetch_signature

        if not (work / "stock_pool.json").exists():
            return "no stock pool"
        sig = prefetch_signature(ep, work, stock_policy_signature())
        return "script/transcript/assets/license policy changed" \
            if state.get("prefetch") != sig else None
    if key == "plan":
        art = work / "scene_plan.json"
        if not art.exists():
            return "no scene plan"
        try:
            if not read_json(art).get("beats"):
                return "scene plan has no beats"
        except json.JSONDecodeError:
            return "scene plan is corrupt"
        up = max(_mtime(work / "transcript.json"), _mtime(work / "local_assets.json"),
                 _mtime(work / "stock_pool.json"), _mtime(work / "semantic_index.json"))
        return "transcript/assets/stock pool newer than plan" if up > _mtime(art) else None
    auth, resolved = _plan_signatures(work)
    if key == "fetch":
        if not (work / "licenses_audit.json").exists():
            return "no license audit"
        if state.get("fetch") != f"{auth}:{stock_policy_signature()}":
            return "plan or stock license policy changed since stock fetch"
        return None
    if key == "cutouts":
        if not (work / "cutouts" / ".ready").exists():
            return "no overlays generated"
        return "plan changed since overlays" if state.get("cutouts") != resolved else None
    if key == "render":
        art = ep / "out" / "final_cut.mp4"
        if not art.exists():
            return "no final cut"
        newest_in = max(_mtime(work / "scene_plan.json"), _mtime(voice),
                        _newest(list((work / "cutouts").glob("*.png"))))
        return "plan/overlays newer than render" if newest_in > _mtime(art) else None
    return None


def stock_policy_signature() -> str:
    """Signature of the current stock license policy (config + available keys)."""
    from src.common import is_offline, load_config
    from src.stage4_fetch_stock import StockFetcher

    return StockFetcher(load_config().get("stock_media", {}), Path("."),
                        offline=is_offline()).policy_signature()


def plan_is_hand_edited(work: Path) -> bool:
    """True if scene_plan.json exists and differs from what Stage 3 last wrote (or that is
    unknown, e.g. a plan from before this check existed) -- regenerating would lose edits."""
    plan_path = work / "scene_plan.json"
    if not plan_path.is_file():
        return False
    try:
        beats = read_json(plan_path).get("beats")
    except json.JSONDecodeError:
        return False  # corrupt: nothing worth protecting
    if not beats:
        return False
    return _load_state(work).get("plan_sig") != _plan_signatures(work)[0]


def _record_state(key: str, work: Path) -> None:
    auth, resolved = _plan_signatures(work)
    state = _load_state(work)
    if key == "prefetch":
        from src.stage_prefetch import prefetch_signature

        state["prefetch"] = prefetch_signature(work.parent, work, stock_policy_signature())
    elif key == "fetch":
        state["fetch"] = f"{auth}:{stock_policy_signature()}"
    elif key == "cutouts":
        state["cutouts"] = resolved
    elif key == "plan":  # a fresh plan invalidates the downstream stamps
        state.pop("fetch", None)
        state.pop("cutouts", None)
        state["plan_sig"] = auth
    else:
        return
    _save_state(work, state)


def _run_log(work: Path, stage: str, seconds: float, status: str, extra: dict[str, Any],
             error: str = "") -> None:
    """Append one JSON line per stage run to work/run_log.jsonl (timings for tuning)."""
    import json

    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage,
           "seconds": round(seconds, 2), "status": status, **{k: v for k, v in extra.items()
                                                              if isinstance(v, (bool, int, str))}}
    if error:
        rec["error"] = error
    try:
        work.mkdir(parents=True, exist_ok=True)
        with (work / "run_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _execute(stage: Stage, episode_id: str, work: Path, **kwargs: Any) -> Any:
    console.print(Panel(f"[bold cyan]Running: {stage.title}[/]", title="Stage"))
    # Stage 4 may downgrade a stock beat to local media (it edits source_id/primary_source).
    # That is the pipeline's own change, not a hand edit: re-stamp the plan signature after it,
    # but only if the plan was untouched before (real hand edits stay protected).
    unedited_before = stage.key == "fetch" and not plan_is_hand_edited(work)
    t0 = time.time()
    try:
        result = _stage_func(stage)(episode_id, work, **kwargs)
        _run_log(work, stage.key, time.time() - t0, "ok", kwargs)
    except Exception as e:
        _run_log(work, stage.key, time.time() - t0, "error", kwargs, str(e)[:300])
        console.print(f"[bold red]Error in {stage.title}:[/]\n  {e}")
        if os.environ.get("VIDEOMAKER_DEBUG"):
            raise
        sys.exit(1)
    if not kwargs.get("draft"):  # a preview must not mark the final render as done
        _record_state(stage.key, work)
    if unedited_before:
        state = _load_state(work)
        state["plan_sig"] = _plan_signatures(work)[0]
        _save_state(work, state)
    console.print(f"[bold green]{stage.title} complete[/]")
    return result


def _transcript_is_synthetic(work: Path) -> bool:
    p = work / "transcript.json"
    try:
        return bool(p.is_file() and read_json(p).get("synthetic"))
    except json.JSONDecodeError:
        return False


def _run_stage(stage: Stage, episode_id: str, force: bool = False) -> Any:
    ep = get_episode_dir(episode_id)
    work = ep / "work"
    reason = "forced" if force else stale_reason(stage.key, ep, work)
    if reason is None:
        console.print(Panel(f"[bold green]Skipping: {stage.title}[/] (up to date)", title="Stage"))
        return None
    kwargs: dict[str, Any] = {}
    if stage.key == "plan" and not force and plan_is_hand_edited(work):
        console.print(
            f"  [yellow]{stage.title}: inputs changed ({reason}) but scene_plan.json was edited "
            "by hand; keeping it. Run `plan <episode> --force` to regenerate (the current plan "
            "is backed up to scene_plan.prev.json).[/]")
        return None
    if stage.key == "transcribe" and force and _transcript_is_synthetic(work):
        console.print("  [dim]Synthetic demo transcript kept (use `transcribe --force`).[/]")
        force = False
    if stage.key == "transcribe":
        kwargs["force"] = force
    console.print(f"  [dim]{stage.title}: {reason}[/]")
    return _execute(stage, episode_id, work, **kwargs)


# --------------------------------------------------------------------------- commands


@app.command()
def run(
    episode_id: str = typer.Argument(..., help="Episode ID (e.g., ep01-sample)"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run every stage"),
    from_stage: str | None = typer.Option(
        None, "--from-stage", help="Skip every stage before this one (audio|transcribe|index|semantic|"
        "prefetch|plan|fetch|cutouts|render)"),
) -> None:
    """Run the full pipeline; stages whose inputs did not change are skipped."""
    if from_stage is not None and from_stage not in STAGE_BY_KEY:
        console.print(f"[red]Unknown stage {from_stage!r}; use one of {', '.join(STAGE_BY_KEY)}[/]")
        raise typer.Exit(1)
    started = from_stage is None
    console.print(Panel("[bold green]Psycho Astronaut Pipeline[/]",
                        subtitle=f"Episode: {episode_id}"))
    for stage in STAGES:
        started = started or stage.key == from_stage
        if started:
            _run_stage(stage, episode_id, force=force)
    ep = get_episode_dir(episode_id)
    console.print(Panel("[bold green]Pipeline complete![/]", subtitle=f"Output: {ep / 'out'}"))


def _single(key: str, episode_id: str, **kwargs: Any) -> None:
    ep = get_episode_dir(episode_id)
    _execute(STAGE_BY_KEY[key], episode_id, ep / "work", **kwargs)


@app.command()
def audio(
    episode_id: str = typer.Argument(..., help="Episode ID"),
    report: bool = typer.Option(False, "--report", help="Print before/after metrics"),
    preset: str = typer.Option(None, "--preset", help="podcast | gentle | aggressive"),
) -> None:
    """Stage 0: clean the voice-over (denoise, EQ, compress, trim silences, -16 LUFS)."""
    _single("audio", episode_id, report=report, preset=preset)


@app.command()
def transcribe(
    episode_id: str = typer.Argument(..., help="Episode ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-transcribe even if up to date"),
) -> None:
    """Stage 1 only: transcribe the voiceover."""
    _single("transcribe", episode_id, force=force)


@app.command()
def index(episode_id: str = typer.Argument(..., help="Episode ID")) -> None:
    """Stage 2 only: index local assets (+ semantic index when a backend is available)."""
    _single("index", episode_id)
    _single("semantic", episode_id)


@app.command()
def stabilize(
    episode_id: str = typer.Argument(..., help="Episode ID (needs work/local_assets.json)"),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore the cache and redo"),
    all_clips: bool = typer.Option(False, "--all", help="Stabilize every clip, not only shaky"),
) -> None:
    """Smooth shaky source clips (vidstab/OpenCV) into work/stabilized/ and update the index."""
    from src.stabilize import apply_to_index

    ep = get_episode_dir(episode_id)
    path = ep / "work" / "local_assets.json"
    if not path.is_file():
        console.print("[red]Run `index` first (no work/local_assets.json).[/]")
        raise typer.Exit(1)
    cfg = load_config()
    if all_clips:
        cfg.setdefault("stabilize", {})["enabled"] = True
    data = read_json(path)
    summary = apply_to_index(data, ep / "work", cfg, log=console.print, force=force)
    write_json(path, data)
    for aid, sc in sorted(summary["scores"].items(), key=lambda kv: -(kv[1] or 0)):
        console.print(f"  shake {sc if sc is not None else '-':>6}  {aid}")
    console.print(f"{len(summary['stabilized'])} stabilized, {summary['skipped']} steady")


@app.command()
def upscale(
    episode_id: str = typer.Argument(..., help="Episode ID (needs work/local_assets.json)"),
    force: bool = typer.Option(False, "--force", "-f", help="Ignore the cache and redo"),
) -> None:
    """Upscale small local photos with Real-ESRGAN (optional extra) and update the asset index."""
    from src.upscale import apply_to_index

    ep = get_episode_dir(episode_id)
    path = ep / "work" / "local_assets.json"
    if not path.is_file():
        console.print("[red]Run `index` first (no work/local_assets.json).[/]")
        raise typer.Exit(1)
    cfg = load_config()
    if force:
        cfg.setdefault("upscale", {})["enabled"] = True
        for f in (ep / "work" / "upscaled").glob("*.png"):
            f.unlink()
    data = read_json(path)
    summary = apply_to_index(data, ep / "work", cfg, log=console.print)
    write_json(path, data)
    console.print(f"{len(summary['upscaled'])} upscaled, {summary['skipped']} skipped")


@app.command()
def semantic(episode_id: str = typer.Argument(..., help="Episode ID")) -> None:
    """Stage 2a only: build the semantic (visual) index of the local assets."""
    _single("semantic", episode_id)


@app.command("eval-matching")
def eval_matching(
    episode_id: str = typer.Argument(..., help="Episode ID (needs work/local_assets.json)"),
    backend: str = typer.Option("all", "--backend", "-b",
                                help="baseline | siglip | ollama | all"),
    pairs: Path | None = typer.Option(None, "--pairs", help="Labelled pairs JSON"),  # noqa: B008
) -> None:
    """Measure top-1/top-3 of sentence -> asset matching against labelled pairs."""
    from src import eval_matching as em

    ep = get_episode_dir(episode_id)
    if not (ep / "work" / "local_assets.json").is_file():
        _single("index", episode_id)
    em.run(ep / "work", backend, pairs)


@app.command()
def prefetch(episode_id: str = typer.Argument(..., help="Episode ID")) -> None:
    """Stage 2b only: pre-fetch a pool of free stock media from the script/transcript."""
    _single("prefetch", episode_id)


@app.command()
def plan(
    episode_id: str = typer.Argument(..., help="Episode ID"),
    force: bool = typer.Option(False, "--force", "-f",
                               help="Overwrite a hand-edited scene_plan.json (backed up first)"),
) -> None:
    """Stage 3 only: (re)generate scene_plan.json."""
    work = get_episode_dir(episode_id) / "work"
    if not force and plan_is_hand_edited(work):
        console.print("[yellow]scene_plan.json was edited by hand; refusing to overwrite it. "
                      "Use `plan --force` (the current plan is saved to scene_plan.prev.json).[/]")
        raise typer.Exit(1)
    _single("plan", episode_id)


@app.command()
def fetch(episode_id: str = typer.Argument(..., help="Episode ID")) -> None:
    """Stage 4 only: fetch CC0/PD stock media for stock beats."""
    _single("fetch", episode_id)


@app.command()
def cutouts(episode_id: str = typer.Argument(..., help="Episode ID")) -> None:
    """Stage 5 only: generate cutouts, badges and lower-thirds."""
    _single("cutouts", episode_id)


@app.command()
def render(episode_id: str = typer.Argument(..., help="Episode ID"),
           draft: bool = typer.Option(False, "--draft", help="Fast low-res preview -> "
                                      "out/draft.mp4 (640x360, x264 ultrafast, no transitions)")
           ) -> None:
    """Re-render from scene_plan.json (re-runs stages 4/5 if the plan changed). Unchanged beats
    come from work/render_cache, so editing one beat re-renders only that beat."""
    for key in ("fetch", "cutouts"):
        stage = STAGE_BY_KEY[key]
        ep = get_episode_dir(episode_id)
        reason = stale_reason(key, ep, ep / "work")
        if reason:
            console.print(f"  [dim]{stage.title}: {reason}[/]")
            _execute(stage, episode_id, ep / "work")
    _single("render", episode_id, draft=True) if draft else _single("render", episode_id)


@app.command()
def create_demo(
    episode_id: str = typer.Argument("ep-demo", help="Episode ID for the demo"),
) -> None:
    """Generate a synthetic engineering episode with dummy assets (offline)."""
    from src.demo_generator import create_demo_episode

    console.print(Panel("[bold cyan]Generating Demo Episode[/]",
                        subtitle="Creating synthetic assets"))
    ep = get_episode_dir(episode_id)
    work = ep / "work"
    create_demo_episode(ep, work)
    console.print(Panel("[bold green]Demo episode created![/]", subtitle=f"Location: {ep}"))


@app.command()
def new(
    episode_id: str | None = typer.Argument(None, help="New episode ID (folder under episodes/)"),
    template: str = typer.Option("blank", "--template", "-t", help="blank | tutorial | diy"),
    theme: str = typer.Option("psycho_terminal", "--theme", help="Theme written to config.yaml"),
    stock: bool = typer.Option(True, "--stock/--no-stock", help="stock_media.enabled in config.yaml"),
    from_ep: str | None = typer.Option(None, "--from", help="Copy config.yaml from another episode"),
    force: bool = typer.Option(False, "--force", help="Fill in missing files of an existing folder"),
    list_templates: bool = typer.Option(False, "--list-templates", help="List templates and exit"),
) -> None:
    """Scaffold a new, empty episode (folders, lean config.yaml, README checklist, sidecar template)."""
    from src import scaffold

    if list_templates:
        for name, desc in scaffold.TEMPLATES.items():
            console.print(f"[bold]{name}[/]  {desc}")
        return
    if not episode_id:
        console.print("[red]Missing episode ID (or use --list-templates)[/]")
        raise typer.Exit(1)
    try:
        res = scaffold.scaffold(episode_id, episodes_root(), template=template, theme=theme, stock=stock,
                                from_ep=from_ep, force=force)
    except (ValueError, OSError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    console.print(f"[green]Created[/] {res.ep} ({len(res.created)} files/folders, {len(res.skipped)} kept)")
    console.print("\n[bold]Next steps[/]")
    console.print(scaffold.next_steps(episode_id), markup=False)


@app.command()
def status(episode_id: str = typer.Argument(..., help="Episode ID")) -> None:
    """Show the status of every pipeline stage for an episode."""
    ep = get_episode_dir(episode_id)
    work = ep / "work"
    artifacts = {
        "audio": work / "voiceover_clean.wav",
        "transcribe": work / "transcript.json",
        "index": work / "local_assets.json",
        "semantic": work / "semantic_index.json",
        "prefetch": work / "stock_pool.json",
        "plan": work / "scene_plan.json",
        "fetch": work / "licenses_audit.json",
        "cutouts": work / "cutouts" / ".ready",
        "render": ep / "out" / "final_cut.mp4",
    }
    table = Table(title=f"Pipeline Status — {episode_id}")
    for col in ("Stage", "Artifact", "Status"):
        table.add_column(col)
    for stage in STAGES:
        path = artifacts[stage.key]
        if not path.exists():
            label = "[yellow]PENDING[/]"
        else:
            reason = stale_reason(stage.key, ep, work)
            label = f"[red]STALE[/] ({reason})" if reason else "[green]DONE[/]"
        table.add_row(stage.title, str(path.relative_to(ep)), label)
    console.print(table)
    from src.scaffold import missing_items

    miss = missing_items(ep)
    if miss:
        console.print("[yellow]Missing:[/]")
        for m in miss:
            console.print(f"  - {m}")
    try:
        assets = read_json(work / "local_assets.json").get("assets", [])
        n_up = sum(1 for a in assets if a.get("upscaled_path"))
        n_img = sum(1 for a in assets if a.get("asset_type") == "image")
        console.print(f"Upscaled photos: {n_up}/{n_img}")
        n_clip = sum(1 for a in assets if a.get("asset_type") == "clip")
        n_st = sum(1 for a in assets if a.get("stabilized_path"))
        console.print(f"Stabilized clips: {n_st}/{n_clip}")
    except (OSError, ValueError):
        pass


@app.command()
def inspect(episode_id: str, stage: str = typer.Argument("plan", help="Stage to inspect")) -> None:
    """Print a stage artifact (transcript, assets, plan, licenses)."""
    work = get_episode_dir(episode_id) / "work"
    stage_files = {
        "transcript": "transcript.json",
        "assets": "local_assets.json",
        "plan": "scene_plan.json",
        "licenses": "licenses_audit.json",
    }
    if stage not in stage_files:
        console.print(f"[red]Unknown stage: {stage}. Available: {', '.join(stage_files)}[/]")
        raise typer.Exit(1)
    path = work / stage_files[stage]
    if not path.exists():
        console.print(f"[red]No data found for stage '{stage}'[/]")
        raise typer.Exit(1)
    console.print(f"[bold cyan]=== {stage.title()} ===[/]")
    console.print(json.dumps(read_json(path), indent=2, ensure_ascii=False), markup=False)


@app.command()
def dashboard(
    episode_id: str = typer.Argument(..., help="Episode ID"),
    theme: str = typer.Option("slate", "--theme", help="slate | psycho"),
    replay: bool = typer.Option(False, "--replay", help="Animate a finished run (for screen recording)"),
    speed: float = typer.Option(1.0, "--speed", min=0.1, help="Replay speed multiplier"),
    follow: bool = typer.Option(False, "--follow", help="Follow a running `run`/`render` (polls files)"),
    video: bool = typer.Option(False, "--video", help="Replay: play final_cut.mp4 as ~4 fps frames in the preview"),
    snapshot: Path | None = typer.Option(  # noqa: B008
        None, "--snapshot", help="Write ONE static frame to out.svg|png|txt and exit"),
    size: str = typer.Option("120x40", "--size", help="Snapshot size COLSxROWS"),
    run_pipeline: bool = typer.Option(False, "--run", help="Run the pipeline as a subprocess, dashboard LIVE"),
    force: bool = typer.Option(False, "--force", help="With --run: re-run every stage"),
    from_stage: str | None = typer.Option(None, "--from-stage", help="With --run: first stage to run"),
    record: Path | None = typer.Option(None, "--record", help="With --run: write an asciicast v2 file"),  # noqa: B008
    snapshot_at_end: Path | None = typer.Option(  # noqa: B008
        None, "--snapshot-at-end", help="With --run: save the final frame (svg|png|txt)"),
    preview: str = typer.Option("auto", "--preview", help="auto|kitty|iterm|sixel|blocks|quadrants"),
) -> None:
    """Terminal dashboard (AGENT PIPELINE) for recording the screen / taking screenshots."""
    from src import dashboard as dash

    if theme not in dash.THEMES:
        console.print(f"[red]Unknown theme {theme!r} (slate|psycho)[/]")
        raise typer.Exit(1)
    th = dash.THEMES[theme]
    if preview not in dash.PREVIEW_MODES:
        console.print(f"[red]--preview must be one of {'|'.join(dash.PREVIEW_MODES)}[/]")
        raise typer.Exit(1)
    # live screen always uses the detected protocol; cast/snapshot frames render with cells (dash.cell_preview)
    dash.set_preview_mode("quadrants" if snapshot is not None and preview == "auto" else preview)
    ep = get_episode_dir(episode_id)
    try:
        from src.common import load_config
        cfg = load_config()
    except (OSError, ValueError):
        cfg = None
    if snapshot is not None:
        try:
            cols, rows = (int(x) for x in size.lower().split("x"))
        except ValueError:
            console.print("[red]--size must look like 120x40[/]")
            raise typer.Exit(1) from None
        d = dash.load_data(ep, episode_id, cfg)
        v = dash.static_view(d, dash.ensure_frames(d))
        console.print(f"[green]Snapshot:[/] {dash.export_snapshot(v, th, snapshot, cols, rows)}")
        return
    if run_pipeline:
        code = dash.run_dashboard(ep, episode_id, cfg, th, force=force, from_stage=from_stage, record=record,
                                  snapshot_at_end=snapshot_at_end)
        raise typer.Exit(code)
    dash.run_live(ep, episode_id, cfg, th, replay=replay, speed=speed, follow=follow, video=video)


if __name__ == "__main__":
    app()
