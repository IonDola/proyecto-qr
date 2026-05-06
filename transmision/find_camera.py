import cv2

print("Buscando cámaras disponibles...\n")
from cv2_enumerate_cameras import enumerate_cameras

for camera_info in enumerate_cameras():
    print(f"ID: {camera_info.index} | Nombre: {camera_info.name}")

import tkinter as tk
root = tk.Tk()
print(root.winfo_screenwidth(), root.winfo_screenheight())
root.destroy()