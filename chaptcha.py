from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import random
import time
import math
import cv2
import numpy as np
# Global variables - defaults
CAPTCHA_SIZE = (120, 40)  # in pixels
FONT_DIR = 'fonts'
BG_DIR = 'backgrounds'
WORD_LENGTH = 8
DEFAULT_BG_COLOR = (230, 230, 230)  # very bright gray
DEFAULT_FONT = 'Dancing Script.ttf'
DEFAULT_FONT_COLOR = (255, 0, 0) #red
DEFAULT_FONT_SIZE = 44
DEFAULT_SAVE_DIR = 'captchas'
NUM_OF_CAPTCHAS = 100

def directory_init():
    dirs = [FONT_DIR, BG_DIR, DEFAULT_SAVE_DIR]

    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Directory '{d}' created.")
        else:
            print(f"Directory '{d}' already exists.")
def get_save_path(file_name, directory='.'):
    """Returns the save path for the given file name in the specified directory.

    If no directory is specified, the file will be saved in the current working directory.
    """
    return os.path.join(directory, file_name)

def save_string_to_file(string, save_path):
    """Saves the given string to a file at the specified save path."""
    with open(save_path, 'w') as file:
        file.write(string)
def find_last_index(folder, starting_index, extension):
    # Ensure folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Start at the given index
    index = starting_index

    # Loop until an available save index is found
    while True:
        # Construct the file path to check
        file_path = os.path.join(folder, f"{index}{extension}")

        # If the file doesn't exist, return the index
        if not os.path.exists(file_path):
            return index

        # Otherwise, increment the index and try again
        index += 1
def get_file_name(path):
    """Get the file name with extension, or name of the directory from a path."""
    return os.path.basename(path)
def generate_random_word(min_letters=0, max_letters=WORD_LENGTH, include_letters=True, include_numbers=True, include_special_chars=True, include_space=True):
    '''This function uses the random library to generate a random word with the specified parameters, 
    including minimum and maximum length, and character sets to include. 
    It checks that at least one character set is included, and raises a ValueError if not. 
    It also uses the WORD_LENGTH global variable as the default value for max_letters.'''
    
    
    # Check that at least one character type is included
    if not (include_letters or include_numbers or include_special_chars or include_space):
        raise ValueError("At least one character type must be included")
    
    # Define the character sets
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    numbers = '0123456789'
    special_chars = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    space = ' '
    
    # Select the character sets based on the include flags
    char_sets = []
    if include_letters:
        char_sets.append(letters)
    if include_numbers:
        char_sets.append(numbers)
    if include_special_chars:
        char_sets.append(special_chars)
    if include_space:
        char_sets.append(space)
    
    # Generate the random word
    if min_letters == 0:
        word_length = max_letters
    else:
        word_length = random.randint(min_letters, max_letters)
    word = ''.join(random.choice(random.choice(char_sets)) for _ in range(word_length))
    
    return word
def get_random_color():
    """Return a random RGB color tuple."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def generate_captcha_text_image(text, font_file, font_size=DEFAULT_FONT_SIZE, width_height=CAPTCHA_SIZE, font_color=DEFAULT_FONT_COLOR, random_pos=False):
    width, height = width_height
    '''This function now takes additional parameters for height and font_color. 
    The scale variable is calculated as the minimum scale required to fit the image within the given width 
    and height dimensions, or 1 if the image is already smaller than either dimension. 
    The final_width and final_height variables are then calculated based on the calculated scale. 
    The image is cropped to remove any transparent pixels around the edges before being resized, 
    to prevent any bleed around the text. The final_image is then created with the desired width and height, 
    and the image is pasted into the center of it.'''
    print(font_file)
    font = ImageFont.truetype(font_file, font_size)
    text_width, text_height = font.getsize(text)
    
    # Create an image with transparent background
    image = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill=font_color)
    
    # Calculate the necessary scale to fit the image within the given width and height dimensions
    scale = min(width / text_width, height / text_height, 1)
    
    # Calculate the final image dimensions after scaling
    final_width = int(scale * text_width)
    final_height = int(scale * text_height)
    
    # Scale the image to the necessary dimensions while leaving the transparent background
    image = image.crop(image.getbbox())
    image = image.resize((final_width, final_height), resample=Image.LANCZOS)
    
    if random_pos:
        # Randomly position the text within the bounds of the image
        max_x_offset = width - final_width
        max_y_offset = height - final_height
        x_offset = random.randint(0, max_x_offset)
        y_offset = random.randint(0, max_y_offset)
    else:
        # Calculate the position of the text to center it on the image
        x_offset = int((width - final_width) / 2)
        y_offset = int((height - final_height) / 2)
    
    # Create a new image with the desired width and height, and paste the text image into the center
    final_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    final_image.paste(image, (x_offset, y_offset))
    
    return final_image

def generate_background(color_or_path=DEFAULT_BG_COLOR, size=CAPTCHA_SIZE):
    '''This function generates a background image of the specified color or from a file, 
    with the specified size. If a color is given, it generates a plain background of that color. 
    If a file path is given, it cuts out a piece of the image or scales it up to match the desired size.'''
    
    # Check if the color_or_path is a valid file path
    if isinstance(color_or_path, str) and os.path.isfile(color_or_path):
        # Check if the file is a supported image type
        if os.path.splitext(color_or_path)[1] not in ['.jpg', '.jpeg', '.png', '.bmp']:
            raise ValueError("Unsupported image format")
        
        # Open the image file
        img = Image.open(color_or_path)
        
        # Check if the image is smaller than the desired size
        if img.size[0] < size[0] or img.size[1] < size[1]:
            #scale_to_fit(img, size)
            raise ValueError("Image size is too small")
        
        # Calculate the maximum coordinates for the top-left corner of the cropped region
        max_x = img.size[0] - size[0]
        max_y = img.size[1] - size[1]
        
        # Choose a random position for the top-left corner of the cropped region
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Crop the image
        img = img.crop((x, y, x+size[0], y+size[1]))
        
        # Scale the image up to the desired size
        img = img.resize(size)
        
        # Return the cropped and scaled image
        return img
    
    # Otherwise, generate a plain background of the specified color
    else:
        # Check if the provided color is valid
        if not isinstance(color_or_path, tuple) or len(color_or_path) != 3:
            raise ValueError("Invalid color")
        
        # Create a new image with the desired size and color
        img = Image.new('RGB', size, color=color_or_path)
        
        # Return the plain background image
        return img

def get_random_file(path, file_type=None):
    '''This function returns the path to a random file in the specified directory 
    that matches the specified file type. If no file type is specified, any file type 
    will be accepted. The file type can also be a specific file extension.'''
    
    # Check if the specified path is a directory
    if not os.path.isdir(path):
        raise ValueError("Path is not a directory")
    
    # Get a list of all files in the directory
    files = os.listdir(path)
    
    # Filter files based on file type
    if file_type is not None:
        if file_type == 'images':
            # Filter for image file extensions
            valid_exts = ('.jpg', '.png', '.jpeg')
            files = [f for f in files if f.lower().endswith(valid_exts)]
        elif file_type == 'fonts':
            valid_exts = ('.ttf')
            files = [f for f in files if f.lower().endswith(valid_exts)]
        else:
            # Filter for specified file extension
            files = [f for f in files if f.lower().endswith('.'+file_type.lower())]
    
    # Choose a random file from the list of valid files
    if len(files) == 0:
        print(path)
        print(files)
        raise ValueError("No valid files found")
    rand_file = random.choice(files)
    
    # Return the path to the selected file
    return os.path.join(path, rand_file)

def scale_to_fit(image, size):
    """Scale the image proportionally to fit within the specified size."""
    width, height = image.size
    aspect_ratio = width / height
    new_width, new_height = size
    new_aspect_ratio = new_width / new_height
    if aspect_ratio > new_aspect_ratio:
        # image is wider than the new size
        target_width = new_width
        target_height = round(target_width / aspect_ratio)
    else:
        # image is taller than the new size
        target_height = new_height
        target_width = round(target_height * aspect_ratio)
    return image.resize((target_width, target_height))
def wave_image(image):
        # Convert the image to a NumPy array
    img_np = np.array(image)

    # Get the height and width of the image
    h, w = img_np.shape[:2]

    # Set the amount of wave distortion to apply
    wave = 20

    # Define the four corners of the original image
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Define the four corners of the distorted image
    dst_pts = np.float32([
        [0, 0],
        [w + np.random.randint(-wave, wave), np.random.randint(-wave, wave)],
        [np.random.randint(-wave, wave), h + np.random.randint(-wave, wave)],
        [w + np.random.randint(-wave, wave), h + np.random.randint(-wave, wave)]
    ])

    # Get the perspective transform matrix and apply it to the image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted = cv2.warpPerspective(img_np, M, (w, h))

    # Apply wave distortion to the distorted image
    for i in range(h):
        distorted[i] = np.roll(distorted[i], int(wave * np.sin(i / h * 2 * np.pi)))

    # Convert the distorted image back to a PIL Image
    distorted = Image.fromarray(distorted)

    # Resize the distorted image to match the original size
    distorted = distorted.resize((w, h))

    # Copy the alpha channel if it exists
    if image.mode == 'RGBA':
        #distorted.putalpha(image.getchannel('A'))
        pass

    return distorted
def skew_image(image):
    # Convert the image to a NumPy array
    img_np = np.array(image)
    
    # Get the height and width of the image
    h, w = img_np.shape[:2]

    # Set the amount of skew to apply (in pixels)
    skew = 5

    # Define the four corners of the original image
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Define the four corners of the distorted image
    dst_pts = np.float32([
        [0, 0],
        [w + np.random.randint(-skew, skew), np.random.randint(-skew, skew)],
        [np.random.randint(-skew, skew), h + np.random.randint(-skew, skew)],
        [w + np.random.randint(-skew, skew), h + np.random.randint(-skew, skew)]
    ])

    # Get the perspective transform matrix and apply it to the image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted = cv2.warpPerspective(img_np, M, (w, h))

    # Convert the distorted image back to a PIL Image
    distorted = Image.fromarray(distorted)

    # Resize the distorted image to match the original size
    distorted = distorted.resize((w, h))

    # Copy the alpha channel if it exists
    if image.mode == 'RGBA':
        #distorted.putalpha(image.getchannel('A'))
        pass

    return distorted
def ripple_image(image):
    # Convert the image to a NumPy array
    img_np = np.array(image)

    # Define the distortion grid
    rows, cols = img_np.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = x.astype(np.float32) - cols / 2.0
    y = y.astype(np.float32) - rows / 2.0

    # Apply the ripple distortion to the grid
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    ripple = r + 5 * np.sin(2 * np.pi * r / 30 - theta)

    # Map the grid to a new set of coordinates
    map_x = (x + ripple * np.cos(theta)) + cols / 2.0
    map_y = (y + ripple * np.sin(theta)) + rows / 2.0

    # Apply the distortion to the image
    distorted = cv2.remap(img_np, map_x, map_y, cv2.INTER_LINEAR)

    # Convert the distorted image back to a PIL Image
    distorted = Image.fromarray(distorted)

    # Copy the alpha channel if it exists
    #if image.mode == 'RGBA':
        #distorted.putalpha(image.getchannel('A'))

    # Resize the distorted image to match the original size
    distorted = distorted.resize((cols, rows))

    return distorted

def random_distort_image(image, ripple=True, wave=False, skew=False):
    if ripple:
        image = ripple_image(image)
    if wave:
        image = wave_image(image)
    if skew:
        image = skew_image(image)
    return image


def flatten_layers(transparent_layers, background, trim_layers=True):
    # If there is only one transparent layer, put it in a list
    if not isinstance(transparent_layers, list):
        transparent_layers = [transparent_layers]

    # Check and convert modes
    for i, img in enumerate(transparent_layers):
        if img.mode != "RGBA":
            transparent_layers[i] = img.convert("RGBA")

    if background.mode != "RGBA":
        background = background.convert("RGBA")

    background_to_merge = background.copy()
    bg_width, bg_height = background.size

    for layer in transparent_layers:
        # Get the width and height of the layer
        layer_width, layer_height = layer.size

        if not trim_layers:
            layer = scale_to_fit(layer, background_to_merge.size)

        # Calculate the position of the layer to center it on the background
        x = (bg_width - layer_width) // 2
        y = (bg_height - layer_height) // 2

        # Paste the layer onto the background
        background_to_merge.alpha_composite(layer, (x, y))

    return background_to_merge


def make_captchas(number=NUM_OF_CAPTCHAS, dimensions=CAPTCHA_SIZE,
              font_size=DEFAULT_FONT_SIZE, font_color=DEFAULT_FONT_COLOR, random_font_color=False,
              number_of_letters=WORD_LENGTH,
              include_special_chars=False, include_space=True,
              include_letters = True, include_numbers = True,
              distort_text_ripple=True, distort_text_wave=False, distort_text_skew=False,
              random_text_pos = True, margin_percentage = 0,
              destination_dir=DEFAULT_SAVE_DIR, 
              bg_dir = BG_DIR, font_dir=FONT_DIR, random_word_len_min_num_of_letters = 0, 
              starting_name_index=0, save_extension='.png', no_text_file=False):
    
    save_name = find_last_index(destination_dir, starting_index=starting_name_index, extension=save_extension)
    for i in range(number):
        bg_source_file = get_random_file(bg_dir, 'images')
        bg_img = generate_background(bg_source_file, size=dimensions)


        captcha_text = generate_random_word(min_letters=random_word_len_min_num_of_letters, max_letters=number_of_letters,
                                            include_letters=include_letters, include_numbers=include_numbers,
                                            include_special_chars=include_special_chars,
                                            include_space=include_space)

        font_file = get_random_file(font_dir, 'fonts')
        if random_font_color:
            font_color = get_random_color()
        if margin_percentage != 0:
            size_ratio = 1 - margin_percentage
            x,y = dimensions
            x = x * size_ratio
            y = y * size_ratio
            
            text_size = (int(x),int(y))
        else:
            text_size = dimensions
        captcha_text_image = generate_captcha_text_image(captcha_text, font_file,
                                                            font_size=font_size, font_color=font_color,
                                                            width_height=text_size, random_pos=random_text_pos)
        
        captcha_text_image = random_distort_image(captcha_text_image, ripple=distort_text_ripple, wave=distort_text_wave, skew=distort_text_skew)
        


        final_captcha = flatten_layers(captcha_text_image, bg_img, True)
        distorted = distort_text_ripple or distort_text_wave or distort_text_skew
        font_name = get_file_name(font_file)
        
        summary = f"TEXT={captcha_text}" + "\n" 
        summary += f"FNAME={font_name}" + "\n"
        summary += f"FSIZE={font_size}"+"\n"
        summary += f"FCOLOR={font_color}"+"\n" 
        summary += f"DISTORTED={distorted}"+"\n"
        summary += f"IMGSIZE={dimensions}"
        
        print(f"{save_name} \n{summary}")
        img_name = f'{save_name}.png'
        txt_name = f'{save_name}.txt'
        
        img_name = get_save_path(img_name, destination_dir)
        txt_name = get_save_path(txt_name, destination_dir)
        
        final_captcha.save(img_name)
        if not no_text_file:
            save_string_to_file(summary, txt_name)
        save_name += 1




make_captchas(number=100, font_color=(0,0,0), dimensions=(240,80), font_size=50,
              random_font_color=False, distort_text_ripple=False, distort_text_skew=True,
              distort_text_wave=False, margin_percentage=0.2, destination_dir="medium_black_text_1"
              )