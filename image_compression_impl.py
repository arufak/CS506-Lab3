import numpy as np
from PIL import Image

# Function to load and preprocess the image
def load_image(image_path):
    # Load the image using Pillow
    img = Image.open(image_path)
    # Convert the image to a NumPy array
    img_np = np.array(img)
    return img_np

# Function to perform SVD on a single channel of the image matrix
def compress_channel_svd(channel_matrix, rank):
    # Perform SVD on the channel matrix
    U, S, Vt = np.linalg.svd(channel_matrix, full_matrices=False)
    
    # Keep only the top 'rank' singular values
    U_reduced = U[:, :rank]
    S_reduced = np.diag(S[:rank])
    Vt_reduced = Vt[:rank, :]
    
    # Reconstruct the compressed channel matrix
    compressed_channel = np.dot(U_reduced, np.dot(S_reduced, Vt_reduced))
    
    return compressed_channel

# Function to perform SVD for image compression
def image_compression_svd(image_np, rank):
    # Check if the image is grayscale or color image
    if len(image_np.shape) == 2:  # Grayscale
        compressed_img = compress_channel_svd(image_np, rank)
    else:
        # List to store compressed channels
        compressed_channels = []
        
        # Loop over the 3 color channels (RGB)
        for i in range(3):
            channel = image_np[:, :, i]
            compressed_channel = compress_channel_svd(channel, rank)
            compressed_channels.append(compressed_channel)
        
        # Stack the compressed channels back into an RGB image
        compressed_img = np.stack(compressed_channels, axis=2)
        
    # Clip values to ensure they remain in the valid pixel range [0, 255]
    compressed_img = np.clip(compressed_img, 0, 255)
    
    return compressed_img.astype(np.uint8)

# Function to concatenate and save the original and quantized images side by side
def save_result(original_image_np, quantized_image_np, output_path):
    # Convert NumPy arrays back to PIL images
    original_image = Image.fromarray(original_image_np)
    quantized_image = Image.fromarray(quantized_image_np)
    
    # Get dimensions
    width, height = original_image.size
    
    # Create a new image that will hold both the original and quantized images side by side
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste original and quantized images side by side
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(quantized_image, (width, 0))
    
    # Save the combined image
    combined_image.save(output_path)
    
if __name__ == '__main__':
    # Load and process the image
    image_path = 'examples/favorite_image.png'  
    output_path_template = 'examples/compressed_image_rank_{}.png'  # Template for output paths
    image_np = load_image(image_path)

    # List of different rank values to experiment with
    ranks = [8, 10, 20, 50, 100]  # You can add more values as you like
    
    for rank in ranks:
        # Perform image quantization using SVD
        quantized_image_np = image_compression_svd(image_np, rank)

        # Save the original and quantized images side by side with the rank in the filename
        output_path = output_path_template.format(rank)
        save_result(image_np, quantized_image_np, output_path)
        
        print(f"Compressed image with rank {rank} saved to {output_path}")

# To run:
# set FLASK_APP=app.py
# set FLASK_ENV=development
# venv\Scripts\flask run --port 3000