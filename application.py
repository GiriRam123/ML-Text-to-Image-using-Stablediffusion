import tkinter as tk

import customtkinter as ctk
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageTk
from torch import autocast

from authtoken import auth_token

# Create the app
app = ctk.CTk()  # Use CTk() for customtkinter
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Correctly specify the master for CTkEntry and other widgets
prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white")
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)["sample"][0]
    
    image.save('generatedimage.png')
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep a reference to avoid garbage collection

trigger = ctk.CTkButton(master=app, height=40, width=120, text="Generate", text_color="white", fg_color="blue", command=generate)
trigger.place(x=206, y=60)

app.mainloop()
