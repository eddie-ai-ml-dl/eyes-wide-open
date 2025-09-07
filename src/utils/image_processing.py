# src/utils/image_processing.py

from PIL import Image


def pad_to_square_and_resize(image: Image.Image, target_size: int, fill_color=(0, 0, 0)) -> Image.Image:
    """
    Pads an image to a square aspect ratio and then resizes it to a target square dimension.
    Preserves aspect ratio by adding padding bars.

    Args:
        image (PIL.Image.Image): The input image to process.
        target_size (int): The desired square dimension (e.g., 224 for 224x224).
        fill_color (tuple): RGB tuple for the color of the padding bars (default: black).

    Returns:
        PIL.Image.Image: The processed image, square and resized to target_size.
    """
    width, height=image.size
    max_dim=max(width, height)

    # Create a new blank square image
    square_padded_img=Image.new(image.mode, (max_dim, max_dim), fill_color)

    # Calculate paste position to center the original crop
    paste_x=(max_dim-width)//2
    paste_y=(max_dim-height)//2
    square_padded_img.paste(image, (paste_x, paste_y))

    # Resize to target classifier size using a high-quality filter
    final_resized_image=square_padded_img.resize((target_size, target_size), Image.LANCZOS)
    return final_resized_image

# You could add other generic image processing functions here in the future
# e.g., rotate_image, apply_gamma_correction, etc.