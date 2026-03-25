import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging

# Define the source and destination directories
DEFAULT_SRC_DIR = os.environ.get("SHADOW_THUMBNAIL_SRC_DIR", "crops")
DEFAULT_THUMB_DIR = os.environ.get("SHADOW_THUMBNAIL_DST_DIR", "thumbnails")
thumbnail_size = (256, 256)  # Set your desired thumbnail size
brightness_factor = 1.0  # Adjust this value to increase brightness (1.0 = no change)
clip_percent = 5  # Percentage to clip from both ends of the histogram

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def stretch_histogram(band, percent=clip_percent):
    """
    Enhance contrast using histogram stretching with percentile clipping.
    
    Args:
        band: Input numpy array
        percent: Percentage of pixels to clip from both ends of the histogram
    
    Returns:
        Contrast enhanced numpy array
    """
    if band.size == 0:
        return band
        
    # Calculate percentiles
    low_percent = percent
    high_percent = 100 - percent
    
    # Get the values at those percentiles
    low_value = np.percentile(band, low_percent)
    high_value = np.percentile(band, high_percent)
    
    # Clip the image to these values
    clipped = np.clip(band, low_value, high_value)
    
    # Rescale the clipped range to fill 0-255
    if high_value > low_value:
        scaled = ((clipped - low_value) * 255 / (high_value - low_value)).astype(np.uint8)
    else:
        scaled = np.zeros_like(band, dtype=np.uint8)
    
    return scaled

def create_thumbnail(args):
    """Create a thumbnail from a GeoTIFF image and save it."""
    image_path, thumb_path = args
    try:
        with rasterio.open(image_path) as src:
            # Read the first band (single-band image)
            band = src.read(1)

            # Check if the max value of the band is greater than zero
            if np.max(band) == 0:
                logger.warning(f"Skipping {image_path}: max value is zero.")
                return False

            # Apply histogram stretching
            band_stretched = stretch_histogram(band, clip_percent)
            
            # Adjust brightness
            band_brightened = np.clip(band_stretched * brightness_factor, 0, 255).astype(np.uint8)

            # Create an image from the brightened data
            img = Image.fromarray(band_brightened)
            
            # Resize and save the thumbnail
            img.thumbnail(thumbnail_size)
            img.save(thumb_path, "JPEG", quality=95)  # Increased JPEG quality for better results
            return True
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return False

def collect_image_paths(src=DEFAULT_SRC_DIR, dest=DEFAULT_THUMB_DIR):
    """Collect all image paths and their corresponding thumbnail paths."""
    image_pairs = []
    for subdir, _, files in os.walk(src):
        # Create the corresponding directory structure in the destination
        os.makedirs(dest, exist_ok=True)

        # Filter for .tif or .tiff files
        tif_files = [file for file in files if file.endswith(('.tif', '.tiff'))]
        
        if tif_files:
            image_path = os.path.join(subdir, tif_files[0])  # Process only the first .tif image
            thumb_path = os.path.join(dest, os.path.basename(subdir) + '_thumb.jpg')
            image_pairs.append((image_path, thumb_path))
    
    return image_pairs

def process_directory_parallel(src=DEFAULT_SRC_DIR, dest=DEFAULT_THUMB_DIR, num_processes=None):
    """Process the directory to create thumbnails in parallel."""
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    
    logger.info(f"Starting parallel processing with {num_processes} processes")
    logger.info(f"Using histogram stretch with {clip_percent}% clip on both ends")
    
    # Collect all image paths first
    image_pairs = collect_image_paths(src, dest)
    total_images = len(image_pairs)
    logger.info(f"Found {total_images} images to process")

    # Create pool of workers
    with Pool(processes=num_processes) as pool:
        # Process images in parallel with progress bar
        results = list(tqdm(
            pool.imap_unordered(create_thumbnail, image_pairs),
            total=total_images,
            desc="Processing images",
            unit="image"
        ))
    
    # Report summary
    successful = sum(1 for r in results if r)
    failed = total_images - successful
    logger.info(f"Processing complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    process_directory_parallel()
