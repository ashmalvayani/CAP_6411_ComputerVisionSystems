# Wan 2.1 T2V — HAR Video Generation (Assignment 8)

Text-to-Video generation with **Wan 2.1 (T2V-1.3B)** to create a 7-class Human Activity Recognition (HAR) mini-dataset.  
Two prompting regimes are provided—**Simple** and **Varied**—and batch scripts generate **10 clips per class**.

- 📁 **Demo videos:** [`videos_outputs/`](./videos_outputs/)
- 🧾 **Prompts & metadata:** [`metadata_simple.csv`](./metadata_simple.csv) • [`metadata.csv`](./metadata.csv)

---

## ✨ Features

- Runs Wan 2.1 (Diffusers) locally; configurable frames/steps/fps.
- Two caption regimes:
  - **Simple:** minimal, class-focused (e.g., “A man is clapping.”)
  - **Varied:** short subject/wording variations for diversity
- Each clip has a sidecar `.txt` with the exact prompt + params.
- CSV logs for full reproducibility (class, index, prompt, seed, size, etc.).

---

## Repository Layout

Assignments/Assignment_8/Code/
├── har_wan_generate.py # Varied prompts (diversity-focused)
├── har_wan_generate_simple.py # Simple prompts (class-focused)
├── wan_generate_like.py # One-off T2V runner
├── videos_outputs/ # 🎬 Demo MP4s (some embedded below)
├── metadata.csv # prompts + seeds + params (varied)
├── metadata_simple.csv # prompts + seeds + params (simple)
└── requirements.txt # environment versions


---

## Quickstart

```bash
# 1) Create environment
conda create -y -n cvs_ass8 python=3.10
conda activate cvs_ass8
pip install -r requirements.txt

# 2) Point to your local Diffusers-format checkpoint directory
#    (e.g., Wan-AI/Wan2.1-T2V-1.3B-Diffusers mirrored locally)
WAN_DIR=/path/to/Wan2.1-T2V-1.3B-Diffusers

# 3a) One-off generation
python wan_generate_like.py \
  --ckpt_dir $WAN_DIR \
  --size 832*480 \
  --prompt "A young boy walking towards the school" \
  --num_frames 61 --sample_guide_scale 6.0 --fps 24

# 3b) Batch: Simple prompts (10 per class; 7 classes)
python har_wan_generate_simple.py \
  --ckpt_dir $WAN_DIR --size 832*480 \
  --out_root ./videos/har_videos_simple \
  --num_per_class 10 --num_frames 61 \
  --num_inference_steps 32 --guidance_scale 6.0

# 3c) Batch: Varied prompts (diversity-focused)
python har_wan_generate.py \
  --ckpt_dir $WAN_DIR --size 832*480 \
  --out_root ./videos/har_videos \
  --num_per_class 10 --num_frames 61 \
  --num_inference_steps 32 --guidance_scale 6.0
```

---

## Clapping (examples)

<table> <tr> <td width="25%"> <strong>clapping_00</strong><br/> <video src="./videos_outputs/clapping_00.mp4" width="240" controls muted></video><br/> <a href="./videos_outputs/clapping_00.mp4">video</a> · <a href="./videos_outputs/clapping_00.txt">prompt</a> </td> <td width="25%"> <strong>clapping_01</strong><br/> <video src="./videos_outputs/clapping_01.mp4" width="240" controls muted></video><br/> <a href="./videos_outputs/clapping_01.mp4">video</a> · <a href="./videos_outputs/clapping_01.txt">prompt</a> </td> <td width="25%"> <strong>clapping_02</strong><br/> <video src="./videos_outputs/clapping_02.mp4" width="240" controls muted></video><br/> <a href="./videos_outputs/clapping_02.mp4">video</a> · <a href="./videos_outputs/clapping_02.txt">prompt</a> </td> <td width="25%"> <strong>clapping_03</strong><br/> <video src="./videos_outputs/clapping_03.mp4" width="240" controls muted></video><br/> <a href="./videos_outputs/clapping_03.mp4">video</a> </td> </tr> </table>


## 🎬 Demo Gallery (inline videos)

> Tip: If a player doesn’t appear, try opening the README on GitHub desktop (mobile sometimes hides video controls).

### Clapping (4 examples)

<table>
<tr>
<td width="25%">
  <strong>clapping_00</strong><br/>
  <video src="./videos_outputs/clapping_00.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/clapping_00.txt">prompt</a></sub>
</td>
<td width="25%">
  <strong>clapping_01</strong><br/>
  <video src="./videos_outputs/clapping_01.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/clapping_01.txt">prompt</a></sub>
</td>
<td width="25%">
  <strong>clapping_02</strong><br/>
  <video src="./videos_outputs/clapping_02.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/clapping_02.txt">prompt</a></sub>
</td>
<td width="25%">
  <strong>clapping_03</strong><br/>
  <video src="./videos_outputs/clapping_03.mp4" width="240" controls muted playsinline></video>
</td>
</tr>
</table>

### Walking (drop in your files)

<table>
<tr>
<td width="25%">
  <strong>walking_00</strong><br/>
  <video src="./videos_outputs/walking_00.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/walking_00.txt">prompt</a></sub>
</td>
<td width="25%">
  <strong>walking_01</strong><br/>
  <video src="./videos_outputs/walking_01.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/walking_01.txt">prompt</a></sub>
</td>
<td width="25%">
  <strong>walking_reading_00</strong><br/>
  <video src="./videos_outputs/walking_reading_00.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/walking_reading_00.txt">prompt</a></sub>
</td>
<td width="25%">
  <strong>walking_phone_00</strong><br/>
  <video src="./videos_outputs/walking_phone_00.mp4" width="240" controls muted playsinline></video><br/>
  <sub><a href="./videos_outputs/walking_phone_00.txt">prompt</a></sub>
</td>
</tr>
</table>
