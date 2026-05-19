from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etude.data.trajectory_io import finite_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Etude tracking episodes from RP1M-like arrays.")
    parser.add_argument("--rp1m-root", required=True, help="RP1M zarr root or directory of .npz files")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-episodes-per-song", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.rp1m_root)
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"{output_root} exists; pass --force to overwrite/add files")
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if source.suffix == ".zarr":
        rows.extend(_extract_from_zarr(source, episodes_dir, max_songs=args.max_songs, max_episodes_per_song=args.max_episodes_per_song))
    else:
        rows.extend(_extract_from_npz_tree(source, episodes_dir, max_songs=args.max_songs, max_episodes_per_song=args.max_episodes_per_song))

    if not rows:
        raise RuntimeError(f"No Etude episodes could be extracted from {source}")

    pd.DataFrame(rows).to_csv(output_root / "manifest.csv", index=False)
    (output_root / "normalization.json").write_text(json.dumps({}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} Etude episodes to {output_root}")


def _extract_from_npz_tree(
    source: Path,
    episodes_dir: Path,
    *,
    max_songs: int | None,
    max_episodes_per_song: int | None,
) -> list[dict[str, object]]:
    candidates = sorted(source.rglob("*.npz")) if source.is_dir() else []
    if max_songs is not None:
        candidates = candidates[:max_songs]
    if max_episodes_per_song is not None:
        candidates = candidates[: max(len(candidates), max_songs or len(candidates)) * max_episodes_per_song]

    rows: list[dict[str, object]] = []
    for episode_id, path in enumerate(candidates):
        with np.load(path, allow_pickle=False) as npz:
            q = np.asarray(npz["q"] if "q" in npz else npz["q_ref"], dtype=np.float32)
            qdot = np.asarray(npz["qdot"] if "qdot" in npz else finite_difference(q, 0.005), dtype=np.float32)
            actions = np.asarray(npz["actions"], dtype=np.float32)
            target_keys = (
                np.asarray(npz["target_keys"], dtype=np.float32)
                if "target_keys" in npz
                else np.zeros((q.shape[0], 88), dtype=np.float32)
            )
            fingertips = _flatten_fingertips(npz["fingertips"]) if "fingertips" in npz else np.zeros((q.shape[0], 30), dtype=np.float32)
        rows.append(
            _write_episode(
                episodes_dir,
                episode_id,
                q=q,
                qdot=qdot,
                actions=actions,
                target_keys=target_keys,
                fingertips=fingertips,
                source_label=str(path),
            )
        )
    return rows


def _extract_from_zarr(
    source: Path,
    episodes_dir: Path,
    *,
    max_songs: int | None,
    max_episodes_per_song: int | None,
) -> list[dict[str, object]]:
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError("zarr is required to extract Etude episodes from a .zarr source") from exc

    root = zarr.open(str(source), mode="r")
    song_names = [name for name in root.keys() if hasattr(root[name], "keys")]
    if max_songs is not None:
        song_names = song_names[:max_songs]

    rows: list[dict[str, object]] = []
    episode_id = 0
    for song_name in song_names:
        group = root[song_name]
        if "hand_joints" not in group or "actions" not in group:
            continue
        num_episodes = int(group["hand_joints"].shape[0])
        if max_episodes_per_song is not None:
            num_episodes = min(num_episodes, max_episodes_per_song)
        for episode_index in range(num_episodes):
            q = np.asarray(group["hand_joints"][episode_index], dtype=np.float32)
            if q.ndim != 2:
                continue
            q = _fit_dim(q, 46)
            qdot = (
                np.asarray(group["joint_velocities"][episode_index], dtype=np.float32)
                if "joint_velocities" in group
                else finite_difference(q, 0.005)
            )
            qdot = _fit_dim(qdot, 46)
            actions = np.asarray(group["actions"][episode_index], dtype=np.float32)
            target_keys = _extract_target_keys(group, episode_index, q.shape[0])
            fingertips = _extract_fingertips(group, episode_index, q.shape[0])
            rows.append(
                _write_episode(
                    episodes_dir,
                    episode_id,
                    q=q,
                    qdot=qdot,
                    actions=actions,
                    target_keys=target_keys,
                    fingertips=fingertips,
                    source_label=f"{source}:{song_name}[{episode_index}]",
                )
            )
            episode_id += 1
    return rows


def _write_episode(
    episodes_dir: Path,
    episode_id: int,
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    actions: np.ndarray,
    target_keys: np.ndarray,
    fingertips: np.ndarray,
    source_label: str,
) -> dict[str, object]:
    q_ref = q.astype(np.float32)
    qdot_ref = finite_difference(q_ref, 0.005)
    out_name = f"episode_{episode_id:06d}.npz"
    np.savez_compressed(
        episodes_dir / out_name,
        q=q.astype(np.float32),
        qdot=qdot.astype(np.float32),
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        actions=actions.astype(np.float32),
        target_keys=target_keys.astype(np.float32),
        fingertips=fingertips.astype(np.float32),
        dt=np.asarray(0.005, dtype=np.float32),
    )
    return {
        "episode_id": episode_id,
        "path": f"episodes/{out_name}",
        "source": source_label,
        "timesteps": int(q.shape[0]),
    }


def _extract_target_keys(group, episode_index: int, steps: int) -> np.ndarray:
    if "goals" not in group:
        return np.zeros((steps, 88), dtype=np.float32)
    goals = np.asarray(group["goals"][episode_index], dtype=np.float32)
    if goals.ndim != 2:
        return np.zeros((steps, 88), dtype=np.float32)
    keys = goals[:, :88]
    if keys.shape[0] != steps:
        keys = _align_steps(keys, steps)
    return keys.astype(np.float32)


def _extract_fingertips(group, episode_index: int, steps: int) -> np.ndarray:
    if "hand_fingertips" not in group:
        return np.zeros((steps, 30), dtype=np.float32)
    fingertips = _flatten_fingertips(np.asarray(group["hand_fingertips"][episode_index], dtype=np.float32))
    if fingertips.shape[0] != steps:
        fingertips = _align_steps(fingertips, steps)
    return fingertips.astype(np.float32)


def _flatten_fingertips(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        return array.reshape(array.shape[0], -1).astype(np.float32)
    if array.ndim == 2:
        return array.astype(np.float32)
    raise ValueError(f"Unsupported fingertip array shape: {array.shape}")


def _fit_dim(value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [T, D] array, got {array.shape}")
    if array.shape[1] == width:
        return array
    output = np.zeros((array.shape[0], width), dtype=np.float32)
    copy_dim = min(width, array.shape[1])
    output[:, :copy_dim] = array[:, :copy_dim]
    return output


def _align_steps(value: np.ndarray, steps: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[0] == steps:
        return array
    if array.shape[0] == 0:
        return np.zeros((steps, array.shape[1]), dtype=np.float32)
    if array.shape[0] > steps:
        return array[:steps]
    pad = np.repeat(array[-1:], steps - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0).astype(np.float32)


if __name__ == "__main__":
    main()
