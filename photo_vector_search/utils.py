import subprocess
import platform
import os

def open_image(image_path):
    """Open an image file with the default image viewer."""
    if platform.system() == "Windows":
        os.startfile(image_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", image_path])
    else:  # Linux and other Unix
        subprocess.run(["xdg-open", image_path])
