import argparse
import math
from datetime import timedelta
from pathlib import Path

import cv2
from faster_whisper import WhisperModel
from PIL import Image, ImageDraw, ImageFont


def fmt_time(seconds):
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def fmt_label(seconds):
    return str(timedelta(seconds=int(seconds)))


def write_transcripts(segments, output_dir):
    txt_path = output_dir / "transcript.txt"
    srt_path = output_dir / "transcript.srt"

    with txt_path.open("w", encoding="utf-8") as txt:
        for segment in segments:
            txt.write(f"[{fmt_label(segment.start)} - {fmt_label(segment.end)}] {segment.text.strip()}\n")

    with srt_path.open("w", encoding="utf-8") as srt:
        for i, segment in enumerate(segments, start=1):
            srt.write(f"{i}\n")
            srt.write(f"{fmt_time(segment.start)} --> {fmt_time(segment.end)}\n")
            srt.write(f"{segment.text.strip()}\n\n")


def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"no se pudo abrir el video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        raise RuntimeError("no se pudo leer el fps del video")

    return frames / fps


def save_frame(video_path, second, output_path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    cv2.imwrite(str(output_path), frame)
    return output_path


def extract_frames(video_path, output_dir, frame_count):
    if frame_count <= 0:
        return []

    duration = get_video_duration(video_path)
    step = duration / (frame_count + 1)
    frame_paths = []

    for i in range(1, frame_count + 1):
        second = step * i
        frame_path = output_dir / f"frame_{second / 60:.1f}m.jpg"
        saved = save_frame(video_path, second, frame_path)
        if saved:
            frame_paths.append(saved)

    return frame_paths


def make_contact_sheet(frame_paths, output_dir):
    if not frame_paths:
        return

    thumbs = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        image.thumbnail((360, 200))
        thumbs.append((frame_path, image.copy()))

    cols = 3
    rows = math.ceil(len(thumbs) / cols)
    cell_w = 380
    cell_h = 240
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for idx, (frame_path, image) in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        sheet.paste(image, (x + 10, y + 10))
        draw.text((x + 10, y + 215), frame_path.name, fill="black", font=font)

    sheet.save(output_dir / "contact_sheet.jpg", quality=90)


def parse_args():
    parser = argparse.ArgumentParser(description="Transcribe vídeos con IA y genera subtítulos y capturas")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="es")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = video_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    segments_iter, _ = model.transcribe(str(video_path), language=args.language, vad_filter=True)
    segments = list(segments_iter)

    write_transcripts(segments, output_dir)
    frame_paths = extract_frames(video_path, output_dir, args.frames)
    make_contact_sheet(frame_paths, output_dir)

    print(f"listo {output_dir}")


if __name__ == "__main__":
    main()

