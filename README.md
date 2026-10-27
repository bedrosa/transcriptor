# Transcriptor de vídeos con IA

Herramienta local para transcribir vídeos mediante el modelo de IA Whisper, utilizando `faster-whisper`. Genera el texto de la transcripción, subtítulos SRT y capturas del vídeo con una hoja de contacto.

## Requisitos

- Python 3.9 o superior y las dependencias de `requirements.txt`.
- Conexión a Internet la primera vez que se utiliza un modelo, para descargarlo automáticamente. El modelo `small` se usa por defecto.
- En Windows, [Microsoft Visual C++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe), si no está instalado.

No es necesario instalar FFmpeg por separado: `faster-whisper` utiliza PyAV, que incluye sus bibliotecas.

### GPU NVIDIA: opción recomendada

Para acelerar la transcripción se recomienda una GPU NVIDIA compatible con CUDA. Antes de ejecutar el programa, instala:

1. El [controlador NVIDIA](https://www.nvidia.com/Download/index.aspx) compatible con tu tarjeta.
2. [CUDA 12.x](https://developer.nvidia.com/cuda-toolkit-archive), que incluye cuBLAS.
3. [cuDNN 9 para CUDA 12](https://developer.nvidia.com/cudnn-downloads).

En Windows, añade al `PATH` las carpetas `bin` de CUDA y cuDNN, y abre una terminal nueva. Estas bibliotecas y el controlador se instalan aparte; `pip install -r requirements.txt` instala las dependencias de Python.

Para comprobar que el controlador reconoce la GPU:

```powershell
nvidia-smi
```

Consulta los [requisitos de GPU de faster-whisper](https://github.com/SYSTRAN/faster-whisper#gpu) si utilizas otras versiones de CUDA o cuDNN.

## Preparar el entorno

Desde la carpeta del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Transcribir un vídeo con GPU

```powershell
python transcribe_video.py "videos/ejemplo.mp4" --device cuda --compute-type float16
```

Para elegir la carpeta de salida:

```powershell
python transcribe_video.py "videos/ejemplo.mp4" --output-dir "salida/ejemplo" --device cuda --compute-type float16
```

El archivo `run_example.ps1` incluye un ejemplo preparado para GPU.

Si no indicas `--device`, el valor `auto` prioriza una GPU NVIDIA cuando la detecta y utiliza CPU si no hay una disponible. Este modo no instala las bibliotecas necesarias para GPU.

## Alternativa con CPU

También puede ejecutarse sin GPU NVIDIA ni CUDA/cuDNN, aunque la velocidad dependerá del procesador y del modelo:

```powershell
python transcribe_video.py "videos/ejemplo.mp4" --device cpu --compute-type int8
```

## Opciones útiles

```powershell
python transcribe_video.py "videos/ejemplo.mp4" --model small --language es --device cuda --compute-type float16
python transcribe_video.py "videos/ejemplo.mp4" --frames 10 --device cuda --compute-type float16
```

## Archivos generados

```text
transcript.txt
transcript.srt
frame_*.jpg
contact_sheet.jpg
```
