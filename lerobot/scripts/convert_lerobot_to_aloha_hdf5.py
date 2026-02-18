"""
Script to convert LeRobot dataset format back to Aloha hdf5 format.

Example usage:

python lerobot/scripts/convert_lerobot_to_aloha_hdf5.py --lerobot-dir data/lerobot/my_repo-test_dataset --output-dir /path/to/output --repo-id my_repo/test_dataset

"""

import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import tyro
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.video_utils import decode_video_frames_torchvision


def extract_camera_name(key: str) -> str:
    """Extract camera name from observation.images.{camera_name} key."""
    if not key.startswith("observation.images."):
        raise ValueError(f"Expected key to start with 'observation.images.', got: {key}")
    return key[len("observation.images.") :]


def convert_lerobot_to_aloha_hdf5(
    lerobot_dir: Path,
    output_dir: Path,
    repo_id: str | None = None,
    *,
    episodes: list[int] | None = None,
    compress_images: bool = False,
    video_backend: str = "pyav",
):
    """
    Convert LeRobot dataset format back to Aloha hdf5 format.
    
    Args:
        lerobot_dir: Directory containing the LeRobot dataset (e.g., "data/lerobot/my_repo-test_dataset")
        output_dir: Directory to save the output hdf5 files
        repo_id: Repository identifier (e.g., "my_repo/test_dataset"). If None, will be inferred from directory name.
        episodes: Optional list of episode indices to convert (default: all episodes)
        compress_images: Whether to compress images using JPEG (default: False, saves uncompressed)
        video_backend: Backend for video decoding ("pyav" or "video_reader", default: "pyav")
    """
    # Robustify when paths are str instead of Path
    lerobot_dir = Path(lerobot_dir)
    output_dir = Path(output_dir)
    
    if not lerobot_dir.exists():
        raise ValueError(f"lerobot_dir does not exist: {lerobot_dir}")
    
    # Infer repo_id from directory name if not provided
    if repo_id is None:
        # Try to infer from directory name (e.g., "my_repo-test_dataset" -> "my_repo/test_dataset")
        dir_name = lerobot_dir.name
        if "-" in dir_name:
            parts = dir_name.split("-", 1)
            repo_id = f"{parts[0]}/{parts[1]}"
            print(f"Inferred repo_id from directory name: {repo_id}")
        else:
            raise ValueError(
                f"Could not infer repo_id from directory name '{dir_name}'. "
                "Please provide --repo-id explicitly (e.g., 'my_repo/test_dataset')."
            )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load LeRobot dataset
    print(f"Loading LeRobot dataset from: {lerobot_dir}")
    # The root should be the parent of lerobot_dir (e.g., if lerobot_dir is "data/lerobot/my_repo-test_dataset",
    # then root should be "data/lerobot" and repo_id should be "my_repo/test_dataset")
    root = lerobot_dir.parent
    dataset = LeRobotDataset(repo_id=repo_id, root=root, split="train", video_backend=video_backend)
    
    print(f"Dataset info:")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  Number of samples: {dataset.num_samples}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Video format: {dataset.video}")
    print(f"  Camera keys: {dataset.camera_keys}")
    
    # Get episode indices to convert
    ep_ids = episodes if episodes is not None else range(dataset.num_episodes)
    
    # Get camera names from camera keys
    camera_names = [extract_camera_name(key) for key in dataset.camera_keys]
    print(f"  Camera names: {camera_names}")
    
    # Check for optional fields
    has_velocity = "observation.velocity" in dataset.hf_dataset.features
    has_effort = "observation.effort" in dataset.hf_dataset.features
    print(f"  Has velocity: {has_velocity}")
    print(f"  Has effort: {has_effort}")
    
    # Process each episode
    for ep_idx in tqdm.tqdm(ep_ids, desc="Converting episodes"):
        # Get episode frame indices
        from_idx = dataset.episode_data_index["from"][ep_idx].item()
        to_idx = dataset.episode_data_index["to"][ep_idx].item()
        num_frames = to_idx - from_idx + 1
        
        # Load non-image data for this episode (avoid decoding videos here, do it separately)
        episode_data = {}
        
        for frame_idx in range(from_idx, to_idx + 1):
            # Use hf_dataset directly to avoid video decoding at this stage
            item = dataset.hf_dataset[frame_idx]
            
            # Collect non-image data for each frame
            for key, value in item.items():
                # Skip image keys - we'll handle them separately
                if key in dataset.camera_keys:
                    continue
                if key not in episode_data:
                    episode_data[key] = []
                # Convert to tensor if needed
                if isinstance(value, (list, np.ndarray)):
                    episode_data[key].append(torch.tensor(value))
                else:
                    episode_data[key].append(value)
        
        # Convert lists to tensors/arrays
        state = torch.stack(episode_data["observation.state"])  # (num_frames, state_dim)
        action = torch.stack(episode_data["action"])  # (num_frames, action_dim)
        
        if has_velocity:
            velocity = torch.stack(episode_data["observation.velocity"])  # (num_frames, state_dim)
        if has_effort:
            effort = torch.stack(episode_data["observation.effort"])  # (num_frames, state_dim)
        
        # Process images for each camera
        images_per_camera = {}
        for camera_key in dataset.camera_keys:
            camera_name = extract_camera_name(camera_key)
            
            # Get video frame references or images
            if dataset.video:
                # Decode frames from video
                # Collect all timestamps for this camera in this episode
                timestamps = []
                video_path = None
                for frame_idx in range(from_idx, to_idx + 1):
                    item = dataset.hf_dataset[frame_idx]
                    video_frame_ref = item[camera_key]
                    
                    # Decode frame from video
                    # video_frame_ref["path"] is relative to the dataset root (e.g., "videos/episode_0.mp4")
                    # Following the same logic as load_from_videos: data_dir = videos_dir.parent
                    video_path_rel = video_frame_ref["path"]
                    data_dir = dataset.videos_dir.parent
                    current_video_path = data_dir / video_path_rel
                    
                    if video_path is None:
                        video_path = current_video_path
                    elif video_path != current_video_path:
                        # All frames should be from the same video file
                        warnings.warn(
                            f"Episode {ep_idx} camera {camera_name} has frames from different video files. "
                            f"This is unexpected but will be handled frame by frame."
                        )
                    
                    timestamps.append(video_frame_ref["timestamp"])
                
                # Decode all frames at once (more efficient)
                if video_path is not None and video_path.exists():
                    frames_tensor = decode_video_frames_torchvision(
                        video_path,
                        timestamps,
                        dataset.tolerance_s,
                        video_backend,
                    )  # (num_frames, C, H, W) in float32 [0,1]
                else:
                    # Fallback: decode frame by frame if paths differ
                    video_frames = []
                    for frame_idx in range(from_idx, to_idx + 1):
                        item = dataset.hf_dataset[frame_idx]
                        video_frame_ref = item[camera_key]
                        video_path_rel = video_frame_ref["path"]
                        data_dir = dataset.videos_dir.parent
                        video_path = data_dir / video_path_rel
                        timestamp = video_frame_ref["timestamp"]
                        
                        frame = decode_video_frames_torchvision(
                            video_path,
                            [timestamp],
                            dataset.tolerance_s,
                            video_backend,
                        )[0]
                        video_frames.append(frame)
                    
                    frames_tensor = torch.stack(video_frames)
            else:
                # Images are stored directly (not as videos)
                # Load images from hf_dataset
                image_list = []
                for frame_idx in range(from_idx, to_idx + 1):
                    item = dataset.hf_dataset[frame_idx]
                    img = item[camera_key]
                    
                    # Convert PIL Image to tensor if needed
                    if not isinstance(img, torch.Tensor):
                        from torchvision import transforms
                        to_tensor = transforms.ToTensor()
                        img = to_tensor(img)
                    
                    image_list.append(img)
                
                frames_tensor = torch.stack(image_list)
            
            # Convert from torch format (float32, channel first, [0,1]) to numpy format (uint8, channel last, [0,255])
            # frames_tensor: (num_frames, C, H, W) in float32 [0,1]
            frames_np = frames_tensor.permute(0, 2, 3, 1).contiguous()  # (num_frames, H, W, C)
            frames_np = (frames_np * 255).clamp(0, 255).byte().numpy()  # (num_frames, H, W, C) in uint8
            
            images_per_camera[camera_name] = frames_np
        
        # Save to hdf5 file
        hdf5_path = output_dir / f"episode_{ep_idx:06d}.hdf5"
        
        with h5py.File(hdf5_path, "w") as f:
            # Save action
            f.create_dataset("/action", data=action.numpy(), compression="gzip")
            
            # Create observations group
            obs_group = f.create_group("/observations")
            
            # Save qpos (state)
            obs_group.create_dataset("qpos", data=state.numpy(), compression="gzip")
            
            # Save optional velocity
            if has_velocity:
                obs_group.create_dataset("qvel", data=velocity.numpy(), compression="gzip")
            
            # Save optional effort
            if has_effort:
                obs_group.create_dataset("effort", data=effort.numpy(), compression="gzip")
            
            # Save images
            images_group = obs_group.create_group("images")
            for camera_name, images in images_per_camera.items():
                if compress_images:
                    # Compress images using JPEG
                    import cv2
                    
                    # Create variable-length dataset for compressed images
                    dt = h5py.special_dtype(vlen=np.dtype("uint8"))
                    compressed_images = images_group.create_dataset(
                        camera_name, (num_frames,), dtype=dt, compression="gzip"
                    )
                    
                    for i in range(num_frames):
                        # Encode image as JPEG
                        # cv2.imencode expects BGR format, but images are in RGB
                        image_bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                        _, encoded = cv2.imencode(".jpg", image_bgr, encode_param)
                        compressed_images[i] = encoded.tobytes()
                else:
                    # Save uncompressed images
                    images_group.create_dataset(
                        camera_name, data=images, compression="gzip"
                    )
        
        print(f"Saved episode {ep_idx} to {hdf5_path}")
    
    print(f"\n✓ Conversion complete! Saved {len(ep_ids)} episodes to: {output_dir}")


if __name__ == "__main__":
    tyro.cli(convert_lerobot_to_aloha_hdf5)
