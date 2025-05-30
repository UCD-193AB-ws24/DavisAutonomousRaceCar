import cv2
import numpy as np

def flood_fill_black(image, x, y, new_color=[255, 255, 255]):
    """
    Flood fill all connected black pixels starting from (x,y) coordinate.
    Treats border pixels as valid neighbors for flood filling.
    
    Args:
        image: Input image (numpy array)
        x: Starting x coordinate
        y: Starting y coordinate
        new_color: Color to replace black pixels with (default: [255, 255, 255] - white)
    """
    print(f"Starting flood fill operation from point ({x}, {y})...")
    
    # Create a copy of the image to avoid modifying the original
    filled_image = image.copy()
    
    # Get image dimensions
    height, width = filled_image.shape[:2]
    total_pixels = height * width
    print(f"Image dimensions: {width}x{height}")
    
    # Check if starting point is within image bounds
    if x < 0 or x >= width or y < 0 or y >= height:
        print(f"Starting point ({x}, {y}) is out of bounds!")
        return filled_image
    
    # Check if starting point is black
    if not np.array_equal(filled_image[y, x], [0, 0, 0]):
        print(f"Starting point ({x}, {y}) is not black! Color: {filled_image[y, x]}")
        return filled_image
    
    print(f"Starting point ({x}, {y}) is black, beginning flood fill...")
    
    # Create a visited array to track processed pixels
    visited = np.zeros((height, width), dtype=bool)
    
    # Create a queue for flood fill
    queue = [(x, y)]
    visited[y, x] = True
    
    # Define 8-connectivity (all surrounding pixels)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    # Progress tracking
    processed_pixels = 0
    last_progress = 0
    
    while queue:
        current_x, current_y = queue.pop(0)
        
        # Change the current pixel to new color
        filled_image[current_y, current_x] = new_color
        processed_pixels += 1
        
        # Report progress every 5%
        progress = int((processed_pixels / total_pixels) * 100)
        if progress > last_progress and progress % 5 == 0:
            print(f"Progress: {progress}% ({processed_pixels}/{total_pixels} pixels processed)")
            last_progress = progress
        
        # Check neighboring pixels
        for dx, dy in directions:
            new_x = current_x + dx
            new_y = current_y + dy
            
            # Check if the new coordinates are within bounds
            if 0 <= new_x < width and 0 <= new_y < height:
                # Check if the new coordinates are not visited and are black
                if not visited[new_y, new_x] and np.array_equal(filled_image[new_y, new_x], [0, 0, 0]):
                    queue.append((new_x, new_y))
                    visited[new_y, new_x] = True
    
    print(f"Flood fill completed! Processed {processed_pixels} pixels.")
    return filled_image

def find_point_inside_track(image):
    """
    Find a pixel coordinate that lies inside the race track.
    Uses state machine to find point between white lines:
    - First white line
    - Black region
    - Second white line
    """
    print("Searching for point inside track...")
    height, width = image.shape[:2]
    
    # Define white color
    white = [255, 255, 255]
    
    # Step size for faster scanning (check every 5th pixel)
    step = 5
    
    # Start scanning from origin (0,0)
    for y in range(0, height, step):
        # Reset state for each new row
        first_white = False
        black_region = False
        last_black_x = 0
        
        for x in range(0, width, step):
            # State machine logic
            if np.array_equal(image[y, x], white):
                if first_white and black_region:
                    # We found the second white line, return the last black point
                    print(f"Found point inside track at: ({last_black_x}, {y})")
                    return (last_black_x, y)
                elif not first_white:
                    first_white = True
                    print(f"Found first white pixel at: ({x}, {y})")
            elif np.array_equal(image[y, x], [0, 0, 0]):  # Only check for black pixels (not gray)
                if first_white and not black_region:
                    black_region = True
                    last_black_x = x
                    print(f"Entered black region at: ({x}, {y})")
    
    print("No suitable point inside track found")
    return None

def convert_white_to_light_gray(image):
    """
    Convert all white pixels to light gray (hex #CDCDCD).
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        numpy array: Image with white pixels converted to light gray
    """
    print("Converting white pixels to light gray...")
    # Create a copy of the image
    result = image.copy()
    
    # Define white color
    white = [255, 255, 255]
    # Define light gray color (hex #CDCDCD)
    light_gray = [205, 205, 205]
    
    # Create a mask for white pixels
    white_mask = np.all(result == white, axis=-1)
    
    # Convert white pixels to light gray
    result[white_mask] = light_gray
    
    return result

def convert_black_to_light_gray(image):
    """
    Convert all black pixels to light gray (hex #CDCDCD).
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        numpy array: Image with black pixels converted to light gray
    """
    print("Converting black pixels to light gray...")
    # Create a copy of the image
    result = image.copy()
    
    # Define black color
    black = [0, 0, 0]
    # Define light gray color (hex #CDCDCD)
    light_gray = [205, 205, 205]
    
    # Create a mask for black pixels
    black_mask = np.all(result == black, axis=-1)
    
    # Convert black pixels to light gray
    result[black_mask] = light_gray
    
    return result

def resize_image(image, target_width):
    """
    Resize image to target width while maintaining aspect ratio and preserving exact pixel values.
    
    Args:
        image: Input image (numpy array)
        target_width: Target width in pixels
        
    Returns:
        numpy array: Resized image
    """
    print(f"Resizing image to width {target_width}...")
    # Get original dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions while maintaining aspect ratio
    aspect_ratio = height / width
    new_width = target_width
    new_height = int(new_width * aspect_ratio)
    
    # Resize using nearest neighbor interpolation to preserve exact pixel values
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    return resized

def convert_border_gray_to_black(image):
    """
    Convert gray pixels that border white pixels to black.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        numpy array: Image with border gray pixels converted to black
    """
    print("Converting border gray pixels to black...")
    # Create a copy of the image
    result = image.copy()
    
    # Define colors
    white = [255, 255, 255]
    gray = [205, 205, 205]
    black = [0, 0, 0]
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create a mask for gray pixels
    gray_mask = np.all(result == gray, axis=-1)
    
    # Define 8-connectivity (all surrounding pixels)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0), (1, 1)]
    
    # For each gray pixel, check if it borders a white pixel
    for y in range(height):
        for x in range(width):
            if gray_mask[y, x]:
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if np.array_equal(result[ny, nx], white):
                            result[y, x] = black
                            break
    
    return result

def add_padding(image, padding_width):
    """
    Add padding of specified width to all edges of the image.
    
    Args:
        image: Input image (numpy array)
        padding_width: Width of padding in pixels
        
    Returns:
        numpy array: Image with padding added
    """
    print(f"Adding {padding_width} pixel padding...")
    # Define the padding color (hex #CDCDCD in BGR)
    padding_color = [205, 205, 205]  # BGR format
    
    # Add padding to all sides
    padded_image = cv2.copyMakeBorder(
        image,
        padding_width,  # top
        padding_width,  # bottom
        padding_width,  # left
        padding_width,  # right
        cv2.BORDER_CONSTANT,
        value=padding_color
    )
    
    return padded_image

def expand_track(image, padding_pixels):
    """
    Expand the track outward by converting black pixels to white if they don't have any gray pixels
    in their 8 neighboring directions.
    Stops the entire expansion process if any black pixel reaches the image border.
    Args:
        image: Input image (numpy array)
        padding_pixels: Number of pixels to expand the track
    Returns:
        numpy array: Image with expanded track
    """
    print(f"Expanding track by {padding_pixels} pixels...")
    result = image.copy()
    white = [255, 255, 255]
    gray = [205, 205, 205]
    black = [0, 0, 0]
    height, width = image.shape[:2]
    
    # Check all 8 neighboring directions
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    for iteration in range(padding_pixels):
        current = result.copy()
        # First pass: convert black pixels to white if they don't have any gray neighbors
        to_white = []
        for y in range(height):
            for x in range(width):
                if np.array_equal(current[y, x], black):
                    # Check if this black pixel is at the edge
                    if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                        print(f"Stopping expansion at iteration {iteration + 1} - black pixel reached image border")
                        return result  # Stop the entire expansion process
                    
                    # Check if any neighbor is gray
                    has_gray_neighbor = False
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if np.array_equal(current[ny, nx], gray):
                                has_gray_neighbor = True
                                break
                    
                    # If no gray neighbors, mark for conversion to white
                    if not has_gray_neighbor:
                        to_white.append((y, x))
        
        for y, x in to_white:
            result[y, x] = white
            
        # Second pass: convert gray pixels bordering white to black
        current = result.copy()
        to_black = []
        for y in range(height):
            for x in range(width):
                if np.array_equal(current[y, x], gray):
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if np.array_equal(current[ny, nx], white):
                                to_black.append((y, x))
                                break
        
        for y, x in to_black:
            result[y, x] = black
            
    return result

def filter_blobs(image):
    """
    Remove black pixels that don't have any gray pixels in their 8 neighboring directions.
    This helps clean up isolated black pixels or small blobs that aren't connected to the main track.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        numpy array: Image with isolated black pixels removed
    """
    print("Filtering out isolated black pixels...")
    result = image.copy()
    white = [255, 255, 255]
    gray = [205, 205, 205]
    black = [0, 0, 0]
    height, width = image.shape[:2]
    
    # Check all 8 neighboring directions
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]
    
    # Find black pixels without gray neighbors
    to_white = []
    for y in range(height):
        for x in range(width):
            if np.array_equal(result[y, x], black):
                # Check if any neighbor is gray
                has_gray_neighbor = False
                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if np.array_equal(result[ny, nx], gray):
                            has_gray_neighbor = True
                            break
                
                # If no gray neighbors, mark for conversion to white
                if not has_gray_neighbor:
                    to_white.append((y, x))
    
    # Convert marked pixels to white
    for y, x in to_white:
        result[y, x] = white
    
    return result 