# EdgeTutor AI — Model Guide

## LLM Models

EdgeTutor uses GGUF-format models via llama-cpp-python. The default is
Phi-3 Mini 4K Instruct Q4, which offers the best balance of quality and
speed for the Jetson Orin Nano 8GB.

### Recommended Models by Board

| Board | RAM | Recommended Model | Size | GPU Layers | Context |
|-------|-----|------------------|------|------------|---------|
| Orin Nano 8GB | 8 GB | Phi-3 Mini Q4 | 2.3 GB | 20 | 2048 |
| Orin Nano 4GB | 4 GB | TinyLlama 1.1B Q8 | 1.1 GB | 10 | 1024 |
| Orin NX 8GB | 8 GB | Phi-3 Mini Q4 | 2.3 GB | 25 | 2048 |
| Orin NX 16GB | 16 GB | Phi-3 Mini Q4 | 2.3 GB | 33 | 4096 |
| AGX Orin 32GB | 32 GB | Phi-3 Mini Q4* | 2.3 GB | 33 | 4096 |
| AGX Orin 64GB | 64 GB | Phi-3 Mini Q4* | 2.3 GB | 33 | 4096 |
| Xavier NX | 8 GB | TinyLlama 1.1B Q8 | 1.1 GB | 10 | 1024 |
| Jetson Nano | 4 GB | TinyLlama 1.1B Q4 | 0.67 GB | 5 | 512 |

*AGX Orin boards can run much larger models (7B-13B). See "Advanced" below.

### How to Change the Model

1. Download a GGUF model file
2. Place it in the `models/` directory
3. Update `.env`:
   ```
   LLM_MODEL_PATH=models/your-model-name.gguf
   ```
4. Adjust GPU layers and context size as needed:
   ```
   LLM_N_GPU_LAYERS=20
   LLM_CONTEXT_SIZE=2048
   ```
5. Restart EdgeTutor

### Downloading Models

Use the built-in script:
```bash
./scripts/download_model.sh
```

Or download manually from Hugging Face:
- [Phi-3 Mini GGUF](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf)
- [TinyLlama GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

### Quantization Guide

| Quantization | Size Reduction | Quality | Speed | Recommended For |
|-------------|---------------|---------|-------|-----------------|
| Q8_0 | ~50% | Excellent | Medium | 16GB+ boards |
| Q6_K | ~58% | Very Good | Medium-Fast | 16GB+ boards |
| Q5_K_M | ~63% | Good | Fast | 8GB boards |
| Q4_K_M | ~70% | Good | Fast | 8GB boards (default) |
| Q4_0 | ~75% | Decent | Very Fast | 4GB boards |
| Q3_K_M | ~78% | Acceptable | Very Fast | 4GB boards |
| Q2_K | ~85% | Low | Fastest | Last resort |

Rule of thumb: Use Q4_K_M for most Jetson boards. Use Q8_0 if you have
plenty of RAM. Go lower (Q3/Q2) only if nothing else fits.

### GPU Layer Offloading

Jetson boards use unified memory (CPU and GPU share RAM). Moving model
layers to GPU accelerates inference but uses more shared memory.

Guidelines:
- **4 GB RAM**: 5-10 layers
- **8 GB RAM**: 15-25 layers
- **16 GB RAM**: 25-33 layers
- **32+ GB RAM**: All layers (-1)

Set in `.env`:
```
LLM_N_GPU_LAYERS=20
```

Or use `-1` to offload everything to GPU.

### Advanced: Larger Models for AGX Orin

If you have an AGX Orin 32/64GB, you can run larger models:

| Model | Params | Q4 Size | Quality |
|-------|--------|---------|---------|
| Llama 3 8B | 8B | ~4.7 GB | Excellent |
| Mistral 7B | 7B | ~4.1 GB | Excellent |
| Phi-3 Small | 7B | ~4.0 GB | Very Good |
| Gemma 2B | 2B | ~1.5 GB | Good |

Download from Hugging Face and update `LLM_MODEL_PATH`.

---

## STT Models (Whisper)

faster-whisper supports these model sizes:

| Model | Size | Speed (Orin Nano) | Quality | Recommended |
|-------|------|-------------------|---------|-------------|
| tiny | ~75 MB | Very fast | Low | 4GB boards |
| base | ~140 MB | Fast | Medium | Testing |
| small | ~460 MB | Medium | Good | **8GB default** |
| medium | ~1.5 GB | Slow | Very Good | 16GB+ boards |
| large-v3 | ~3 GB | Very slow | Excellent | 32GB+ boards |

Set in `.env`:
```
STT_MODEL_SIZE=small
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
```

For 4GB boards, use:
```
STT_MODEL_SIZE=tiny
STT_COMPUTE_TYPE=int8
```

---

## TTS Voices (Piper)

Piper TTS uses ONNX voice models. Download voices from:
https://rhasspy.github.io/piper-samples/

### Default Voice
- **en_US-lessac-medium**: Clear American English, natural sounding

### Adding New Voices

1. Download the `.onnx` and `.onnx.json` files
2. Place both in the `voices/` directory
3. Update `.env`:
   ```
   TTS_VOICE_MODEL=voices/your-voice-name.onnx
   ```
4. Restart EdgeTutor

### Available Voices (English)

| Voice | Quality | Size | Notes |
|-------|---------|------|-------|
| en_US-lessac-medium | Medium | ~75 MB | Default, clear |
| en_US-lessac-high | High | ~100 MB | Higher quality |
| en_US-amy-medium | Medium | ~75 MB | Female British |
| en_US-ryan-medium | Medium | ~75 MB | Male |
| en_GB-alan-medium | Medium | ~75 MB | British male |

For non-English voices, see the full catalog at the Piper samples page.

---

## Embedding Models (RAG)

The RAG system uses sentence-transformers for embeddings:

| Model | Dimensions | Size | Quality |
|-------|-----------|------|---------|
| all-MiniLM-L6-v2 | 384 | ~80 MB | **Default** - good balance |
| all-MiniLM-L12-v2 | 384 | ~120 MB | Better quality |
| paraphrase-MiniLM-L3-v2 | 384 | ~60 MB | Faster, lower quality |

Set in `.env`:
```
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

The model is downloaded once during setup/first run and cached locally.
