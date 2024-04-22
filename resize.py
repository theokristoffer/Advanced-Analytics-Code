import os
from PIL import Image

def downscale_images(input_folder, output_folder, scale_factor):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    import os
from PIL import Image

def downscale_images(input_folder, output_folder, scale_factor):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all files in the input folder
    files = os.listdir(input_folder)
    
    for file in files:
        # Check if the file is an image
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg'):
            # Open the image file (with error handling)
            img_path = os.path.join(input_folder, file)
            try:
                img = Image.open(img_path)
                
                # Get the new dimensions
                width, height = img.size
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize the image using LANCZOS resampling filter
                resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, file)
                resized_img.save(output_path)
            except Exception as e:
                print(f"Skipped {file} due to error: {e}")



# Example usage
input_folder = 'input_images'
output_folder = 'output_images'
scale_factor = 0.5  # Change this to the desired scale factor

downscale_images(r"C:\Users\Theo\Documents\Advanced Analytics\images", "C:\\Users\\Theo\\Documents\\Advanced Analytics\\images_downscaled", 0.5)
print("all done")