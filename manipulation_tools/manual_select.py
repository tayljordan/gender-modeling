import os
import shutil
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk

# Directories
source_dir = "/Users/jordantaylor/Desktop/gender-modeling/manipulation_tools/11Nov24_f_faces"
destination_dir = "/Users/jordantaylor/Desktop/gender-modeling/manipulation_tools/11Nov24_f_faces_clean"

# Get list of images
images = [f for f in os.listdir(source_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
current_index = 0

def load_image(index):
    """Load image for the current index."""
    image_path = os.path.join(source_dir, images[index])
    image = Image.open(image_path)
    image.thumbnail((800, 800))  # Resize for display
    return ImageTk.PhotoImage(image)

def show_next_image(event=None):
    """Show the next image."""
    global current_index, image_label, current_image
    if current_index < len(images) - 1:
        current_index += 1
        current_image = load_image(current_index)
        image_label.config(image=current_image)
        update_status()

def show_prev_image(event=None):
    """Show the previous image."""
    global current_index, image_label, current_image
    if current_index > 0:
        current_index -= 1
        current_image = load_image(current_index)
        image_label.config(image=current_image)
        update_status()

def move_image(event=None):
    """Move the current image to the destination directory."""
    global current_index
    image_path = os.path.join(source_dir, images[current_index])
    shutil.move(image_path, destination_dir)
    del images[current_index]
    if current_index >= len(images):
        current_index = len(images) - 1
    if images:
        show_next_image()
    else:
        image_label.config(image="")
        status_label.config(text="No more images.")

def update_status():
    """Update the status label."""
    status_label.config(text=f"Image {current_index + 1} of {len(images)}")

# Create the main window
root = Tk()
root.title("Image Selector")

# Bind keys
root.bind("<Right>", show_next_image)
root.bind("<Left>", show_prev_image)
root.bind("<space>", move_image)

# Load the first image
current_image = load_image(current_index)

# Widgets
image_label = Label(root, image=current_image)
image_label.pack()

status_label = Label(root, text=f"Image {current_index + 1} of {len(images)}")
status_label.pack()

button_frame = Button(root)
button_frame.pack()

prev_button = Button(button_frame, text="Previous", command=show_prev_image)
prev_button.grid(row=0, column=0)

next_button = Button(button_frame, text="Next", command=show_next_image)
next_button.grid(row=0, column=1)

select_button = Button(root, text="Select (Move)", command=move_image)
select_button.pack()

# Start the GUI loop
root.mainloop()