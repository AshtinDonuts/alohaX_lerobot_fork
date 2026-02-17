"""
Script to convert Aloha hdf5 data to the LeRobot dataset format for local use only.

Example usage:
python lerobot/scripts/convert_aloha_data_to_lerobotv1.py --raw-dir /path/to/raw/data --repo-id my_repo/test_dataset --local-dir data/lerobot/my_dataset

i.e.
python lerobot/scripts/convert_aloha_data_to_lerobotv1.py --raw-dir /mnt/c2d9b23a-b03e-4fdb-82ad-59f039ec9e3e/khw/green_ball_ssil/ --repo-id my_repo/test_dataset    --local-dir data/lerobot/my_repo-test_dataset

"""

import shutil
import warnings
from pathlib import Path

import tyro

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.populate_dataset import save_lerobot_dataset_on_disk
from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import from_raw_to_lerobot_format
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id


def port_aloha(
    raw_dir: Path,
    repo_id: str,
    local_dir: Path,
    *,
    episodes: list[int] | None = None,
    fps: int | None = None,
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    force_override: bool = False,
    cache_dir: Path = Path("/tmp"),
    encoding: dict | None = None,
):
    """
    Convert Aloha hdf5 data to LeRobot dataset format for local use only.
    
    Args:
        raw_dir: Directory containing episode_*.hdf5 files
        repo_id: Repository identifier (e.g., "my_repo/test_dataset")
        local_dir: Local directory to save the dataset (e.g., "data/lerobot/my_dataset")
        episodes: Optional list of episode indices to convert
        fps: Frame rate (default: 50 for Aloha)
        video: Whether to encode images as videos (default: True)
        batch_size: Batch size for computing stats
        num_workers: Number of workers for computing stats
        force_override: Whether to override existing local_dir
        cache_dir: Temporary directory for videos/images during conversion
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
    lerobot_dataset.stats = stats

    # Save dataset to disk using the same function as record_eps.py
    save_lerobot_dataset_on_disk(lerobot_dataset)
    print(f"\n✓ Dataset saved to: {local_dir / 'train'}")
    print(f"✓ Metadata saved to: {local_dir / 'meta_data'}")
    if video:
        print(f"✓ Videos saved to: {local_dir / 'videos'}")

    return lerobot_dataset


if __name__ == "__main__":
    tyro.cli(port_aloha)
