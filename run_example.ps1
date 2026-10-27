$video = "videos/ejemplo.mp4"
$out = "salida/ejemplo"

python transcribe_video.py $video --output-dir $out --model small --language es --frames 8 --device cuda --compute-type float16
