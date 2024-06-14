from PIL import Image
import os

def resize_image(image_path, output_path, size=(640, 480)):
    img = Image.open(image_path)
    img_resized = img.resize(size, Image.ANTIALIAS)
    img_resized.save(output_path)
    print(f"Resized {image_path} and saved to {output_path}")

def process_images(root_folder, output_root_folder, size=(640, 480)):
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file == "dbScan_p8.png":
                # Compute relative path to maintain directory structure
                relative_path = os.path.relpath(root, root_folder)
                # Create corresponding directory in the output folder
                output_dir = os.path.join(output_root_folder, relative_path)
                os.makedirs(output_dir, exist_ok=True)
                # Formulate the input and output paths
                img_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file)
                # Resize and save the image
                resize_image(img_path, output_path, size)

# Get the directory where the script is located
root_folder = os.path.dirname(os.path.abspath(__file__))
output_root_folder = os.path.join(root_folder, 'wynik')

process_images(root_folder, output_root_folder)
