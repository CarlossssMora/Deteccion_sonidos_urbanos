import os
import numpy as np
import pandas as pd
import customtkinter as ctk
import pygame
from tkinter import messagebox
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ======================================================
# Rutas del proyecto
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "mel_metadata.csv")
AUDIO_DIR = os.path.join(BASE_DIR, "data")
SPEC_DIR = os.path.join(BASE_DIR, "espectrogramas")
MODEL_PATH = os.path.join(BASE_DIR, "crnn_urbansound8k.h5")

# ======================================================
# Clases
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
# Cargar datos y modelo
# ======================================================
df = pd.read_csv(CSV_PATH)
df["spec_file"] = df["spectrogram"].apply(extract_filename)
df["audio_file"] = df["spec_file"].str.replace(".npy", ".wav", regex=False)
df["display_name"] = df["audio_file"]

model = load_model(MODEL_PATH, compile=False)

# ======================================================
# Audio
# ======================================================
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# ======================================================
# UI config
# ======================================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Clasificador")
app.geometry("1100x650")
app.resizable(False, False)

# ======================================================
# CONTENEDOR PRINCIPAL
# ======================================================
container = ctk.CTkFrame(app, corner_radius=24, fg_color="#0f172a")
container.pack(padx=20, pady=20, fill="both", expand=True)

# ======================================================
# HEADER
# ======================================================
title = ctk.CTkLabel(
    container,
    text="Clasificador de Sonidos Urbanos",
    font=ctk.CTkFont(size=26, weight="bold"),
    text_color="#e5e7eb"
)
title.pack(pady=(20, 25))

# ======================================================
# LAYOUT EN DOS COLUMNAS
# ======================================================
content = ctk.CTkFrame(container, fg_color="transparent")
content.pack(fill="both", expand=True, padx=10)

left = ctk.CTkFrame(content, fg_color="transparent")
left.pack(side="left", fill="y", padx=(0, 10))

right = ctk.CTkFrame(content, fg_color="transparent")
right.pack(side="right", fill="both", expand=True)

# ======================================================
# CARD AUDIO
# ======================================================
card_audio = ctk.CTkFrame(left, corner_radius=18, fg_color="#111827")
card_audio.pack(pady=(0, 20), fill="x")

ctk.CTkLabel(card_audio, text="🎧 Audio",
             font=ctk.CTkFont(size=16, weight="bold"),
             text_color="#e5e7eb").pack(anchor="w", padx=20, pady=(15, 8))

combo = ctk.CTkComboBox(
    card_audio,
    values=df["display_name"].tolist(),
    width=320
)
combo.pack(padx=20, pady=10)
combo.set(df["display_name"].iloc[0])

def play_audio():
    row = df[df["display_name"] == combo.get()].iloc[0]
    path = find_audio_in_folds(AUDIO_DIR, row["audio_file"])
    if path:
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
    else:
        messagebox.showerror("Error", "Audio no encontrado")

def stop_audio():
    pygame.mixer.music.stop()

btns = ctk.CTkFrame(card_audio, fg_color="transparent")
btns.pack(pady=(5, 15))

ctk.CTkButton(btns, text="▶ Reproducir", width=140,
              fg_color="#2563eb", hover_color="#3b82f6",
              command=play_audio).grid(row=0, column=0, padx=8)

ctk.CTkButton(btns, text="⏹ Detener", width=140,
              fg_color="#020617", hover_color="#111827",
              border_width=1, border_color="#1f2937",
              command=stop_audio).grid(row=0, column=1, padx=8)

# ======================================================
# CARD RESULTADOS
# ======================================================
card_result = ctk.CTkFrame(left, corner_radius=18, fg_color="#111827")
card_result.pack(fill="x")

ctk.CTkLabel(card_result, text="📊 Clasificación",
             font=ctk.CTkFont(size=16, weight="bold"),
             text_color="#e5e7eb").pack(anchor="w", padx=20, pady=(15, 8))

result_label = ctk.CTkLabel(
    card_result, text="—",
    font=ctk.CTkFont(size=22, weight="bold"),
    text_color="#93c5fd"
)
result_label.pack(pady=(10, 4))

top3_label = ctk.CTkLabel(
    card_result, text="Top-3:\n—",
    font=ctk.CTkFont(size=13),
    text_color="#9ca3af",
    justify="left"
)
top3_label.pack(pady=(0, 10))

# ======================================================
# CARD ESPECTROGRAMA
# ======================================================
card_spec = ctk.CTkFrame(right, corner_radius=18, fg_color="#111827")
card_spec.pack(fill="both", expand=True)

ctk.CTkLabel(card_spec, text="📈 Espectrograma",
             font=ctk.CTkFont(size=16, weight="bold"),
             text_color="#e5e7eb").pack(anchor="w", padx=20, pady=(15, 8))

spec_canvas_frame = ctk.CTkFrame(card_spec, fg_color="transparent")
spec_canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)

canvas = None

def show_spectrogram(spec):
    global canvas
    for widget in spec_canvas_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(spec.squeeze(), aspect="auto", origin="lower", cmap="magma")
    ax.axis("off")

    canvas = FigureCanvasTkAgg(fig, master=spec_canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# ======================================================
# CLASIFICACIÓN
# ======================================================
def classify():
    row = df[df["display_name"] == combo.get()].iloc[0]
    spec_path = os.path.join(SPEC_DIR, row["spec_file"])

    spec = np.load(spec_path)
    show_spectrogram(spec)

    if spec.ndim == 2:
        spec = spec[..., np.newaxis]
    spec = spec[np.newaxis, ...]

    pred = model.predict(spec, verbose=0)[0]

    top3 = np.argsort(pred)[-3:][::-1]
    result_label.configure(text=CLASS_NAMES[top3[0]])

    top3_text = "\n".join(
        f"{CLASS_NAMES[i]}: {pred[i]*100:.1f}%"
        for i in top3
    )
    top3_label.configure(text=f"Top-3:\n{top3_text}")

ctk.CTkButton(
    card_result,
    text="Clasificar sonido",
    width=260, height=44,
    fg_color="#2563eb", hover_color="#3b82f6",
    command=classify
).pack(pady=(10, 20))

# ======================================================
# Ejecutar
# ======================================================
app.mainloop()
