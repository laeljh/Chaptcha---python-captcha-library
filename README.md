# Chaptcha---python-captcha-library
A python library for creating random customized captcha. Supports backgrounds from file cut-outs or plain color, supports custom fonts and sizes, and many other options.

This library is perfect for creating solved captcha datasets. Allows to select letters, special, digits, spaces and set up and configure most of the possible options. 
It can output image + text summary pair. 
I've created it to make datasets for AI training, but it will work great as a regular captcha generator.


# Usage 
1) pip install dependencies OpenCV2, PIL, os, numpy, random, math
2) put the script in the desired folder
3) call directory init to create directories
  - put background images into backgrounds -> the script will cut out pieces of these to make backgorunds, make sure they're larger than requested background size
  - put ttf fonts into fonts -> the script will randomly select a font each time, make sure to check all the characters are avaialble (not a demo font)
4) call the function providing parameters or editing global variables first:
  def make_captchas(number=NUM_OF_CAPTCHAS, dimensions=CAPTCHA_SIZE,
              font_size=DEFAULT_FONT_SIZE, font_color=DEFAULT_FONT_COLOR, random_font_color=False,
              number_of_letters=WORD_LENGTH,
              include_special_chars=False, include_space=True,
              include_letters = True, include_numbers = True,
              distort_text_ripple=True, distort_text_wave=False, distort_text_skew=False,
              random_text_pos = True, margin_percentage = 0,
              destination_dir=DEFAULT_SAVE_DIR, 
              bg_dir = BG_DIR, font_dir=FONT_DIR, random_word_len_min_num_of_letters = 0, 
              starting_name_index=0, save_extension='.png', no_text_file=False)
              
5) if you want to use my code, remember to attribute me and cosnider donating to some open-source project!
