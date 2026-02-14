# EdgeTutor AI — Complete Hardware Setup Guide

A step-by-step guide to building your EdgeTutor AI appliance from scratch,
including unboxing, assembly, OS installation, and first boot.

---

## Table of Contents

1. [Recommended Hardware](#recommended-hardware)
2. [Shopping List with Links](#shopping-list)
3. [Unboxing & Assembly](#unboxing--assembly)
4. [Flashing JetPack OS](#flashing-jetpack-os)
5. [First Boot & Configuration](#first-boot--configuration)
6. [Installing EdgeTutor AI](#installing-edgetutor-ai)
7. [Peripheral Setup](#peripheral-setup)
8. [Troubleshooting](#troubleshooting)

---

## Recommended Hardware

### Primary: Jetson Orin Nano 8GB (Best for EdgeTutor)

The **NVIDIA Jetson Orin Nano 8GB Developer Kit** is the recommended platform.
It offers the best price-to-performance ratio for running local AI models:

- 8 GB unified (shared CPU/GPU) LPDDR5 memory
- 1024 CUDA cores (Ampere architecture)
- 40 TOPS AI performance
- Supports JetPack 6.x (Ubuntu 22.04)
- ~$249 MSRP

### Alternative Boards

| Board | RAM | CUDA Cores | Price | EdgeTutor Rating |
|-------|-----|-----------|-------|-----------------|
| Orin Nano 8GB | 8 GB | 1024 | ~$249 | ⭐⭐⭐⭐⭐ Best value |
| Orin NX 16GB | 16 GB | 2048 | ~$599 | ⭐⭐⭐⭐⭐ Premium |
| Orin Nano 4GB | 4 GB | 512 | ~$199 | ⭐⭐⭐ Limited |
| AGX Orin 32GB | 32 GB | 2048 | ~$999 | ⭐⭐⭐⭐⭐ Overkill |
| Xavier NX | 8 GB | 384 | ~$399 | ⭐⭐⭐ Older |
| Jetson Nano (orig) | 4 GB | 128 | ~$149 | ⭐⭐ Minimal |

---

## Shopping List

### Essential Items

#### 1. NVIDIA Jetson Orin Nano Developer Kit (8GB)
The brain of your EdgeTutor.

🛒 **Amazon**: [NVIDIA Jetson Orin Nano Developer Kit](https://www.amazon.com/dp/B0BZJTQ5YP)
- Includes: Orin Nano module + reference carrier board
- Price: ~$249

#### 2. MicroSD Card (128GB+ recommended)
For the operating system and EdgeTutor files. Get a fast one.

🛒 **Amazon**: [Samsung EVO Select 128GB microSD](https://www.amazon.com/dp/B09B1HMJ9Z)
- A2 rated, UHS-I U3, up to 130 MB/s
- Price: ~$13

🛒 **Alternative**: [SanDisk Extreme 256GB microSD](https://www.amazon.com/dp/B09X7BK27V)
- A2 rated, up to 160 MB/s (faster)
- Price: ~$25

#### 3. NVMe SSD (Optional but HIGHLY recommended)
Dramatically faster than microSD. Models load much faster from NVMe.

🛒 **Amazon**: [Samsung 970 EVO Plus 250GB NVMe](https://www.amazon.com/dp/B08V83JZH4)
- M.2 2230/2242 compatible with Orin Nano
- Price: ~$40

🛒 **Alternative**: [WD Blue SN570 500GB NVMe](https://www.amazon.com/dp/B09HKG6SDF)
- More storage for models and content
- Price: ~$45

> **Note**: Check your carrier board's M.2 slot size. The Orin Nano Dev Kit
> supports M.2 Key M 2230 NVMe.

#### 4. Power Supply (USB-C, 5V/3A minimum)
The Orin Nano needs a reliable USB-C power supply.

🛒 **Amazon**: [Noctua USB-C PD Power Supply 65W](https://www.amazon.com/dp/B0BJLHP1C2)
- USB-C PD, 65W, reliable
- Price: ~$25

🛒 **Alternative**: [Official NVIDIA Power Supply](https://www.amazon.com/dp/B0CQKFH11V)
- The official option if available
- Price: ~$15

> **Important**: Use a quality power supply. Cheap ones cause instability.

#### 5. USB Keyboard & Mouse
For initial setup. Any USB keyboard and mouse will work.

🛒 **Amazon**: [Logitech MK270 Wireless Keyboard & Mouse](https://www.amazon.com/dp/B079JLY5M5)
- Wireless combo with USB receiver
- Price: ~$23

#### 6. HDMI Monitor or Display (for setup)
Any HDMI monitor works. You only need it for initial OS setup —
after that, you can use EdgeTutor headlessly via the web UI.

#### 7. Ethernet Cable or WiFi Adapter
For initial setup (downloading packages and models).

The Orin Nano Dev Kit has a built-in Ethernet port. WiFi requires
an M.2 WiFi module or USB WiFi adapter.

🛒 **Amazon**: [Intel AX200 WiFi 6 M.2 Module](https://www.amazon.com/dp/B07TLBNZ9M)
- WiFi 6 + Bluetooth 5.0
- M.2 Key E (check your carrier board)
- Price: ~$18

🛒 **USB Alternative**: [TP-Link USB WiFi Adapter](https://www.amazon.com/dp/B07P5PRK7J)
- Plug and play, no M.2 needed
- Price: ~$15

### Optional Peripherals (for Voice & Vision)

#### 8. USB Webcam (for worksheet scanning)
Any USB webcam works. For best OCR results, get one with autofocus.

🛒 **Amazon**: [Logitech C920 HD Pro Webcam](https://www.amazon.com/dp/B00829D0GM)
- 1080p, autofocus, good for document capture
- Price: ~$60

🛒 **Budget**: [Logitech C270 HD Webcam](https://www.amazon.com/dp/B004FHO5Y6)
- 720p, fixed focus (still works for OCR)
- Price: ~$20

#### 9. USB Microphone (for voice input)
For push-to-talk voice input.

🛒 **Amazon**: [Blue Snowball iCE USB Microphone](https://www.amazon.com/dp/B014PYGTUQ)
- Clear audio, plug and play
- Price: ~$40

🛒 **Budget**: [FIFINE USB Microphone](https://www.amazon.com/dp/B06XQ39VCH)
- Affordable, good enough for STT
- Price: ~$14

#### 10. Speaker (for TTS audio output)
For the tutor to speak responses aloud.

🛒 **Amazon**: [JBL Go 3 Portable Speaker](https://www.amazon.com/dp/B08KW1KR5G)
- Bluetooth + 3.5mm aux
- Price: ~$30

🛒 **Budget**: [AmazonBasics USB Speaker](https://www.amazon.com/dp/B07DDK3W5Z)
- USB powered, plug and play
- Price: ~$17

#### 11. Case / Enclosure (Optional)
Protect your Jetson and make it look like a proper appliance.

🛒 **Amazon**: [Jetson Orin Nano Case with Fan](https://www.amazon.com/dp/B0CTBJW12H)
- Aluminum case with cooling fan
- Price: ~$20-35

### Total Cost Estimate

| Configuration | Estimated Cost |
|-------------|---------------|
| **Minimal** (Jetson + SD + power) | ~$280 |
| **Recommended** (+ NVMe + WiFi) | ~$340 |
| **Full Kit** (+ camera + mic + speaker) | ~$450 |
| **Premium** (Orin NX 16GB + all accessories) | ~$800 |

---

## Unboxing & Assembly

### What's in the Jetson Orin Nano Developer Kit Box

1. Jetson Orin Nano module (attached to carrier board)
2. Quick Start Guide
3. Power supply or barrel jack (varies by region)

### Assembly Steps

#### Step 1: Inspect the Board
- Remove from anti-static bag carefully
- Check that the Orin Nano module is properly seated on the carrier board
- Locate the key ports:
  - USB-C (power)
  - USB-A ports (3x)
  - HDMI output
  - Ethernet (RJ45)
  - MicroSD slot (under the module)
  - M.2 Key M slot (for NVMe SSD)
  - M.2 Key E slot (for WiFi)

#### Step 2: Install NVMe SSD (if purchased)
1. Locate the M.2 Key M slot on the carrier board
2. Insert the NVMe SSD at a 30° angle
3. Press down gently and secure with the screw
4. The SSD should lie flat against the board

#### Step 3: Install WiFi Module (if purchased)
1. Locate the M.2 Key E slot
2. Insert the WiFi module at a 30° angle
3. Secure with screw
4. Attach the antenna wires (if included)

#### Step 4: Insert MicroSD Card
1. The microSD slot is on the underside of the module
2. Insert the flashed microSD card (see next section)
3. Push until it clicks

#### Step 5: Install in Case (if purchased)
1. Follow the case instructions
2. Ensure proper airflow around the module's heatsink
3. Connect the fan to the fan header (if applicable)

---

## Flashing JetPack OS

### Option A: Flash MicroSD Card (Easiest)

1. **Download JetPack SD Card Image**:
   - Go to: https://developer.nvidia.com/embedded/jetpack
   - Download the SD card image for your board (JetPack 6.x)
   - File will be ~15 GB

2. **Flash with Balena Etcher**:
   - Download Etcher: https://www.balena.io/etcher/
   - Insert microSD into your PC's card reader
   - Open Etcher → Select the JetPack image → Select the SD card → Flash
   - Wait ~10-15 minutes

3. **Insert the flashed SD card into the Jetson**

### Option B: Flash to NVMe SSD (Better Performance)

If you have an NVMe SSD, boot from it instead of the SD card:

1. Flash the SD card first (Option A)
2. Boot the Jetson from the SD card
3. Use NVIDIA SDK Manager or `nvme-migrate` to copy the OS to NVMe
4. Update the boot order to boot from NVMe

> This gives dramatically better read/write speeds for model loading.

---

## First Boot & Configuration

### Step 1: Connect Everything
1. Insert the flashed microSD card (or NVMe)
2. Connect HDMI to monitor
3. Connect USB keyboard and mouse
4. Connect Ethernet cable (recommended) or WiFi adapter
5. Connect power supply (USB-C)

### Step 2: Power On
1. The Jetson boots automatically when power is connected
2. You'll see the NVIDIA logo, then the Ubuntu setup wizard
3. **This first boot takes 2-5 minutes** — be patient

### Step 3: Initial Ubuntu Setup
1. Accept the EULA
2. Choose your language and keyboard layout
3. Select your timezone
4. Create a username and password
   - Recommended: username `edgetutor`
   - Choose a secure password
5. Wait for the setup to complete (~5 minutes)
6. You'll land on the Ubuntu 22.04 desktop

### Step 4: Connect to Internet
- **Ethernet**: Should auto-connect
- **WiFi**: Click the network icon in the top-right → select your network

### Step 5: Update the System
Open a terminal (Ctrl+Alt+T) and run:
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 6: Verify CUDA
```bash
nvcc --version
```
You should see CUDA 12.x. If not, JetPack may not be installed correctly.

---

## Installing EdgeTutor AI

### One-Command Install

```bash
# Clone the repository
git clone https://github.com/thatcooperguy/Nvidia_Jetson_AI_Tutor.git
cd Nvidia_Jetson_AI_Tutor

# Run setup (installs everything, downloads models)
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

The setup script will:
1. Install system packages (Tesseract, FFmpeg, etc.)
2. Create a Python virtual environment
3. Install Python dependencies
4. Compile llama-cpp-python with CUDA support
5. Offer to download the default LLM and TTS models
6. Create the `.env` configuration file

### Start EdgeTutor

```bash
./scripts/run.sh
```

Then open a browser on any device on the same network:
```
http://<jetson-ip-address>:7860
```

To find your Jetson's IP:
```bash
hostname -I
```

### Index Your Content (Optional)

Add PDF/TXT/MD files to `content/` and run:
```bash
./scripts/ingest_content.sh
```

---

## Peripheral Setup

### Camera Setup
1. Plug in USB webcam
2. Verify it's detected:
   ```bash
   ls /dev/video*
   ```
   You should see `/dev/video0`
3. The EdgeTutor UI will show the camera automatically

### Microphone Setup
1. Plug in USB microphone
2. Verify it's detected:
   ```bash
   arecord -l
   ```
3. Set it as default input (if needed):
   ```bash
   pavucontrol  # GUI audio settings
   ```
4. The EdgeTutor UI's push-to-talk will use the default mic

### Speaker Setup
1. Connect via USB, 3.5mm, or HDMI audio
2. Verify output:
   ```bash
   aplay -l
   ```
3. Test:
   ```bash
   speaker-test -t wav -c 2
   ```

---

## Troubleshooting

### Jetson Won't Boot
- **No lights**: Check power supply. Use a quality USB-C PD adapter.
- **Green light but no display**: Try a different HDMI cable. Ensure
  the SD card is fully inserted and properly flashed.
- **Boot loop**: Re-flash the SD card. It may be corrupted.

### Out of Memory Errors
- Reduce `LLM_N_GPU_LAYERS` in `.env` (try 10 or 5)
- Use a smaller model (TinyLlama instead of Phi-3)
- Reduce `LLM_CONTEXT_SIZE` to 1024 or 512
- Close other applications running on the Jetson
- Use `STT_MODEL_SIZE=tiny` instead of `small`

### Slow Performance
- Ensure CUDA is being used: check logs for "cuda" device
- Increase `LLM_N_GPU_LAYERS` (more GPU = faster)
- Use NVMe SSD instead of microSD for model storage
- Ensure the Jetson's power mode is set to max:
  ```bash
  sudo nvpmodel -m 0     # Max performance
  sudo jetson_clocks      # Lock clocks to max
  ```

### Camera Not Detected
- Check `ls /dev/video*`
- Try a different USB port
- Some cameras need `v4l-utils`:
  ```bash
  sudo apt install v4l-utils
  v4l2-ctl --list-devices
  ```

### Microphone Not Working
- Check `arecord -l`
- Ensure the mic is set as default in audio settings
- Try recording a test:
  ```bash
  arecord -d 5 test.wav && aplay test.wav
  ```

### WiFi Issues
- If using USB adapter, check `lsusb` to verify it's detected
- Install drivers if needed:
  ```bash
  sudo apt install linux-firmware
  ```
- For M.2 WiFi, ensure antennas are connected

### Model Download Fails
- Check internet connection
- Use `wget` with `--continue` flag to resume:
  ```bash
  cd models/
  wget --continue <model-url>
  ```
- Verify disk space: `df -h`

### Edge Cases
- **No GPU detected**: Install JetPack properly. Run `nvidia-smi` to check.
- **Tesseract errors**: `sudo apt install tesseract-ocr tesseract-ocr-eng`
- **Audio errors**: `sudo apt install libsndfile1 ffmpeg portaudio19-dev`
