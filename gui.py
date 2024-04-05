import tkinter as tk
from tkinter import ttk, filedialog, Label
from PIL import Image, ImageTk
from functions import *
import joblib
import os

rf_classifier = joblib.load('random_forest_model.pkl')

def upload_image(file_path):
    img = preprocess_image(file_path, 240, 240)
    img = img.reshape(1, -1)
    prediction = rf_classifier.predict(img)
    showimage(file_path)
    butt_display.grid_forget()
    Label(root, text="Result: " + prediction[0], font=("Arial", 20)).grid(row=4, column=0)

def upload_wrapper():
    global file_path
    file_path = filedialog.askopenfilename()
    file_ref_path = os.path.relpath(file_path)
    for widget in root.winfo_children():
        if widget not in [text1, upload_button]:
            widget.grid_forget()
    upload_image(file_ref_path)

def showimage(file_path):
    img = Image.open(file_path)

    max_width = int(screen_width * 0.5)
    max_height = int(screen_height * 0.5)
    original_width, original_height = img.size
    scale_width = max_width / original_width
    scale_height = max_height / original_height
    scale = min(scale_width, scale_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    img_tk = ImageTk.PhotoImage(img)
    picture = Label(root, image=img_tk)
    picture.grid(row=2, column=0)
    picture.image = img_tk

root = tk.Tk()
root.title("Brain Cancer Detector")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
desired_width = screen_width-50
desired_height = screen_height-100
x = (screen_width - desired_width) // 2
y = (screen_height - desired_height) // 2 - 50
root.geometry(f'{desired_width}x{desired_height}+{x}+{y}')

text1 = Label(root, text="Press the Button to Process an Image.", font=("Arial", 20))
text1.grid(row=0, column=0)

style = ttk.Style()
style.configure('TButton', font=('Arial', 14), foreground='black', background='#4CAF50', padding=10)

upload_button = ttk.Button(root, text="Upload Image", command=upload_wrapper, style='TButton')
upload_button.grid(row=1, column=0)

butt_display = tk.Button(root, text="Display Image", command=lambda: showimage(file_path))

root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)

root.mainloop()