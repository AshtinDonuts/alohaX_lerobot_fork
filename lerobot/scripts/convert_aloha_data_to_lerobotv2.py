"""
Script to convert Aloha hdf5 data to the LeRobot dataset format using current HuggingFace methods.

Example usage:
python lerobot/scripts/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id my_repo/test_dataset

Note: This script works with local directories. The raw_dir must exist locally and contain episode_*.hdf5 files.
By default, push_to_hub is False to avoid connecting to HuggingFace.
"""

import json
import shutil
import warnings
from pathlib import Path

import torch
import tyro
from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import from_raw_to_lerobot_format
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.common.datasets.utils import create_branch, create_lerobot_dataset_card, flatten_dict


def save_meta_data(
    info: dict, stats: dict, episode_data_index: dict, meta_data_dir: Path
):
    """Save metadata files (info.json, stats.safetensors, episode_data_index.safetensors)."""
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # save info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)

    # save stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    # save episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)


def push_meta_data_to_hub(repo_id: str, meta_data_dir: str | Path, revision: str | None):
    """Upload metadata files to HuggingFace hub."""
    api = HfApi()
    api.upload_folder(
        folder_path=meta_data_dir,
        path_in_repo="meta_data",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
    )


def push_dataset_card_to_hub(
    repo_id: str, revision: str | None, tags: list | None = None, text: str | None = None
):
    """Creates and pushes a LeRobotDataset Card with appropriate tags to easily find it on the hub."""
    card = create_lerobot_dataset_card(tags=tags, text=text)
    card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=revision)


def push_videos_to_hub(repo_id: str, videos_dir: str | Path, revision: str | None):
    """Upload video files to HuggingFace hub."""
    api = HfApi()
    api.upload_folder(
        folder_path=videos_dir,
        path_in_repo="videos",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns="*.mp4",
    )


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    local_dir: Path | None = None,
    fps: int | None = None,
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    force_override: bool = False,
    cache_dir: Path = Path("/tmp"),
    encoding: dict | None = None,
):
    """
    Convert Aloha hdf5 data to LeRobot dataset format using current HuggingFace methods.
    
    Args:
        raw_dir: Directory containing episode_*.hdf5 files
        repo_id: Repository identifier (e.g., "my_repo/test_dataset")
        episodes: Optional list of episode indices to convert
        push_to_hub: Whether to push to HuggingFace hub
        local_dir: Local directory to save the dataset (e.g., "data/lerobot/my_dataset")
        fps: Frame rate (default: 50 for Aloha)
        video: Whether to encode images as videos (default: True)
        batch_size: Batch size for computing stats
        num_workers: Number of workers for computing stats
        force_override: Whether to override existing local_dir
        cache_dir: Temporary directory for videos/images
        encoding: Video encoding parameters
    """
    check_repo_id(repo_id)
    user_id, dataset_id = repo_id.split("/")

    # Robustify when `raw_dir` is str instead of Path
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise ValueError(
            f"raw_dir does not exist: {raw_dir}. Please provide a valid local directory path."
        )

    if local_dir:
        # Robustify when `local_dir` is str instead of Path
        local_dir = Path(local_dir)

        # Send warning if local_dir isn't well formated
        if local_dir.parts[-2] != user_id or local_dir.parts[-1] != dataset_id:
            warnings.warn(
                f"`local_dir` ({local_dir}) doesn't contain a community or user id `/` the name of the dataset that match the `repo_id` (e.g. 'data/lerobot/pusht'). Following this naming convention is advised, but not mandatory.",
                stacklevel=1,
            )

        # Check we don't override an existing `local_dir` by mistake
        if local_dir.exists():
            if force_override:
                shutil.rmtree(local_dir)
            else:
                raise ValueError(f"`local_dir` already exists ({local_dir}). Use `--force-override`.")

        meta_data_dir = local_dir / "meta_data"
        videos_dir = local_dir / "videos"
    else:
        # Temporary directory used to store images, videos, meta_data
        meta_data_dir = Path(cache_dir) / "meta_data"
        videos_dir = Path(cache_dir) / "videos"

    # Convert dataset from Aloha hdf5 format to LeRobot format
    fmt_kwgs = {
        "raw_dir": raw_dir,
        "videos_dir": videos_dir,
        "fps": fps,
        "video": video,
        "episodes": episodes,
        "encoding": encoding,
    }

    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(**fmt_kwgs)

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    if local_dir:
        hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
        hf_dataset.save_to_disk(str(local_dir / "train"))
        print(f"\n✓ Dataset saved to: {local_dir / 'train'}")
        print(f"✓ Metadata saved to: {local_dir / 'meta_data'}")
        if video:
            print(f"✓ Videos saved to: {local_dir / 'videos'}")

    if push_to_hub or local_dir:
        # mandatory for upload
        save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if push_to_hub:
        print(f"\n📤 Pushing dataset to HuggingFace Hub: {repo_id}")
        hf_dataset.push_to_hub(repo_id, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        push_dataset_card_to_hub(repo_id, revision="main")
        if video:
            push_videos_to_hub(repo_id, videos_dir, revision="main")
        create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)
        print(f"✓ Successfully pushed to: https://huggingface.co/datasets/{repo_id}")

    if not local_dir and not push_to_hub:
        # Clean up temporary files if not saving locally or pushing to hub
        print("\n⚠️  Warning: No output location specified!")
        print("   Use --local-dir to save locally or --push-to-hub to upload to HuggingFace.")
        print(f"   Temporary files in {cache_dir} will be cleaned up.")
        if videos_dir.exists():
            shutil.rmtree(videos_dir)
        if meta_data_dir.exists():
            shutil.rmtree(meta_data_dir)

    return lerobot_dataset


if __name__ == "__main__":
    tyro.cli(port_aloha)
