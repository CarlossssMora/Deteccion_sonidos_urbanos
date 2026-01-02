import os
import numpy as np
import pandas as pd
import customtkinter as ctk
import pygame
from tkinter import messagebox
from tensorflow.keras.models import load_model

# ======================================================
# Rutas del proyecto
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "mel_metadata.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "data")
SPEC_DIR = os.path.join(BASE_DIR, "espectrogramas")
MODEL_PATH = os.path.join(BASE_DIR, "crnn_urbansound8k.h5")

# ======================================================
# Clases UrbanSound8K
# ======================================================
CLASS_NAMES = {
    0: "Air conditioner",
    1: "Car horn",
    2: "Children playing",
    3: "Dog bark",
    4: "Drilling",
    5: "Engine idling",
    6: "Gun shot",
    7: "Jackhammer",
    8: "Siren",
    9: "Street music"
}

# ======================================================
# Utilidades
# ======================================================
def extract_filename(path: str) -> str:
    return path.replace("\\", "/").split("/")[-1]

def find_audio_in_folds(audio_root, audio_filename):
    for fold in os.listdir(audio_root):
        fold_path = os.path.join(audio_root, fold)
        if os.path.isdir(fold_path):
            candidate = os.path.join(fold_path, audio_filename)
            if os.path.exists(candidate):
                return candidate
    return None

# ======================================================
# Cargar CSV y modelo
# ======================================================
df = pd.read_csv(CSV_PATH)
df["spec_file"] = df["spectrogram"].apply(extract_filename)
df["audio_file"] = df["spec_file"].str.replace(".npy", ".wav", regex=False)
df["display_name"] = df["audio_file"]

model = load_model(MODEL_PATH, compile=False)

# ======================================================
# Inicializar audio
# ======================================================
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# ======================================================
# Configuración UI
# ======================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Clasificador")
app.geometry("720x560")
app.resizable(False, False)

# ======================================================
# CONTENEDOR PRINCIPAL
# ======================================================
container = ctk.CTkFrame(
    app,
    corner_radius=24,
    fg_color="#0f172a"
)
container.pack(padx=20, pady=20, fill="both", expand=True)

# ======================================================
# HEADER
# ======================================================
header = ctk.CTkFrame(
    container,
    fg_color="transparent"
)
header.pack(fill="x", pady=(10, 30))

title = ctk.CTkLabel(
    header,
    text="Clasificador de Sonidos Urbanos",
    font=ctk.CTkFont(size=26, weight="bold"),
    text_color="#e5e7eb"
)
title.pack()

# ======================================================
# CARD: SELECCIÓN DE AUDIO
# ======================================================
card_audio = ctk.CTkFrame(
    container,
    corner_radius=18,
    fg_color="#111827",
    border_width=1,
    border_color="#1f2937"
)
card_audio.pack(fill="x", pady=(0, 24))

ctk.CTkLabel(
    card_audio,
    text="🎧 Selección de audio",
    font=ctk.CTkFont(size=16, weight="bold"),
    text_color="#e5e7eb"
).pack(anchor="w", padx=24, pady=(20, 8))

combo = ctk.CTkComboBox(
    card_audio,
    values=df["display_name"].tolist(),
    width=520,
    fg_color="#020617",
    border_color="#1f2937",
    button_color="#1d4ed8",
    button_hover_color="#2563eb"
)
combo.pack(padx=24, pady=10)
combo.set(df["display_name"].iloc[0])

buttons_audio = ctk.CTkFrame(card_audio, fg_color="transparent")
buttons_audio.pack(pady=(8, 20))

def play_selected_audio():
    selected = combo.get()
    row = df[df["display_name"] == selected].iloc[0]
    audio_path = find_audio_in_folds(AUDIO_DIR, row["audio_file"])

    if audio_path is None:
        messagebox.showerror("Audio no encontrado", row["audio_file"])
        return

    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

def stop_audio():
    pygame.mixer.music.stop()

ctk.CTkButton(
    buttons_audio,
    text="▶ Reproducir",
    width=160,
    height=38,
    fg_color="#1d4ed8",
    hover_color="#2563eb",
    corner_radius=12,
    command=play_selected_audio
).grid(row=0, column=0, padx=12)

ctk.CTkButton(
    buttons_audio,
    text="⏹ Detener",
    width=160,
    height=38,
    fg_color="#020617",
    hover_color="#111827",
    border_width=1,
    border_color="#1f2937",
    corner_radius=12,
    command=stop_audio
).grid(row=0, column=1, padx=12)

# ======================================================
# CARD: CLASIFICACIÓN
# ======================================================
card_result = ctk.CTkFrame(
    container,
    corner_radius=18,
    fg_color="#111827",
    border_width=1,
    border_color="#1f2937"
)
card_result.pack(fill="x")

ctk.CTkLabel(
    card_result,
    text="📊 Clasificación",
    font=ctk.CTkFont(size=16, weight="bold"),
    text_color="#e5e7eb"
).pack(anchor="w", padx=24, pady=(20, 10))

result_label = ctk.CTkLabel(
    card_result,
    text="Clase predicha: —",
    font=ctk.CTkFont(size=20, weight="bold"),
    text_color="#93c5fd"
)
result_label.pack(pady=(8, 4))

confidence_label = ctk.CTkLabel(
    card_result,
    text="Confianza: —",
    font=ctk.CTkFont(size=14),
    text_color="#9ca3af"
)
confidence_label.pack(pady=(0, 14))

def classify():
    selected = combo.get()
    row = df[df["display_name"] == selected].iloc[0]
    spec_path = os.path.join(SPEC_DIR, row["spec_file"])

    if not os.path.exists(spec_path):
        messagebox.showerror("Error", "Espectrograma no encontrado")
        return

    spec = np.load(spec_path)
    if spec.ndim == 2:
        spec = spec[..., np.newaxis]
    spec = spec[np.newaxis, ...]

    pred = model.predict(spec, verbose=0)
    cls = int(np.argmax(pred))
    conf = float(np.max(pred))

    result_label.configure(text=f"Clase predicha: {CLASS_NAMES[cls]}")
    confidence_label.configure(text=f"Confianza: {conf:.2%}")

ctk.CTkButton(
    card_result,
    text="Clasificar sonido",
    width=300,
    height=44,
    fg_color="#2563eb",
    hover_color="#3b82f6",
    corner_radius=14,
    command=classify
).pack(pady=(10, 24))

# ======================================================
# Ejecutar app
# ======================================================
app.mainloop()
