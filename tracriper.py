import whisper
import warnings

# Sembunyikan semua peringatan
warnings.filterwarnings("ignore")

# Load model Whisper (pilih ukuran model sesuai kemampuan perangkat)
model = whisper.load_model("base")  # Bisa diganti "small", "medium", atau "large"

# Path file audio (ganti sesuai lokasi file Anda)
audio_path = "Audio/example.mp3"

# Transkripsi file audio (gunakan fp16=False untuk menghindari peringatan tentang FP16)
result = model.transcribe(audio_path, fp16=False)

# Tampilkan hasil transkripsi
print("Hasil Transkripsi:")
print(result["text"])
