import torch
import cv2
import numpy as np

from diffusers import AutoencoderKLCogVideoX

vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.float16).to(
    "cuda")

# Enable optimizations
vae.enable_slicing()
vae.enable_tiling()


# ---- Video Processing Functions ----

def load_video(video_path):
    """Extract frames from a video."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def encode_frames(frames):
    """Encode frames to latent space."""
    latents = []
    with torch.no_grad():
        frames_resized = np.array([cv2.resize(frame, (512, 512)) for frame in frames])  # Resize all frames
        print(f"np array frames shape {frames_resized.shape}")  # np array frames shape (708, 512, 512, 3)
        #  batch_size, num_channels, num_frames, height, width = x.shape
        frames_tensor = torch.tensor(frames_resized).permute(3, 0, 1, 2).unsqueeze(0)  # Convert to (Frames, Channels, H, W)
        frames_tensor = frames_tensor.to(torch.float16).to("cuda") / 127.5 - 1  # Normalize
        print(f"frames_tensor shape {frames_tensor.shape}")
        latent = vae.encode(frames_tensor).latent_dist.sample()
        latents.append(latent)

    return latents


def decode_frames(latents):
    """Decode latents back to video frames."""
    frames = []
    with torch.no_grad():
        for latent in latents:
            decoded = vae.decode(latent).sample
            decoded_image = ((decoded.squeeze(0).permute(1, 2, 0).cpu().float() + 1) * 127.5).clamp(0,
                                                                                                    255).byte().numpy()
            decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
            frames.append(decoded_image)

    return frames


def save_video(frames, output_path, fps=30):
    """Save frames as a video."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


# ---- Main Execution ----

# Load video frames
video_path = "videos/yoga.mp4"
frames = load_video(video_path)
print(f"Loaded {len(frames)} frames.")
frames = frames[:48]

# Encode frames
latents = encode_frames(frames)
print("Encoding complete.")

# Decode frames
# decoded_frames = decode_frames(latents)
# print("Decoding complete.")
#
# # Save output video
# save_video(decoded_frames, "output/output.mp4")
# print("Saved output video as output.mp4.")
