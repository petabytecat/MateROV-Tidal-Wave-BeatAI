import pygame
import sys
import random
import json
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import shutil
import subprocess

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_WIDTH = 1300
WINDOW_HEIGHT = 768
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Marine Species Viewer")

# Colors
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Button dimensions
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 60
BUTTON_MARGIN = 20  # Reduced margin

# Font settings
FONT_SIZE = 32
TITLE_FONT_SIZE = 48
LABEL_FONT_SIZE = 24

button_labels = ["Original Image", "Guess Image", "Labeled Image", "AI", "Random Image"]  # Swapped positions 1 and 2
current_image = None
showing_image = False

# Create button rectangles with adjusted positioning
total_buttons_width = BUTTON_WIDTH * len(button_labels) + BUTTON_MARGIN * (len(button_labels) - 1)
start_x = (WINDOW_WIDTH - total_buttons_width) // 2

buttons = [
    pygame.Rect(
        start_x + (i * (BUTTON_WIDTH + BUTTON_MARGIN)),
        WINDOW_HEIGHT - 100,
        BUTTON_WIDTH,
        BUTTON_HEIGHT
    )
    for i in range(len(button_labels))
]


# Your number_to_animal dictionary remains the same
number_to_animal = {
    1: "Strongylocentrotus fragilis",
    2: "Ophiuroidea",
    3: "Porifera",
    4: "Anoplopoma fimbria",
    5: "Psolus squamatus",
    6: "Actiniaria",
    7: "Sebastolobus",
    8: "Chionoecetes tanneri",
    9: "Keratoisis",
    10: "Asteroidea",
    11: "Heterochone calyx",
    12: "Merluccius productus",
    13: "Rathbunaster californicus",
    14: "Hexactinellida",
    15: "Paragorgia arborea"
}


def add_text_to_surface(surface, text, position, font_size=LABEL_FONT_SIZE, color=RED):
    """Add high-resolution text to a pygame surface"""
    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, color)

    # Add white outline for better visibility
    outline_surface = font.render(text, True, WHITE)
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        surface.blit(outline_surface, (position[0]+dx, position[1]+dy))

    surface.blit(text_surface, position)

def run_yolov9_detection(image_path):
    """Run YOLOv9 detection on the image"""

    ai_detection_folder = os.path.join("temp", "ai_detection")

    # Check if the ai_detection folder exists and delete it safely
    if os.path.exists(ai_detection_folder):
        try:
            # Wait for a short time before deletion, in case the folder is still in use
            time.sleep(0.5)  # Adjust as needed
            shutil.rmtree(ai_detection_folder)
            print(f"Deleted existing folder: {ai_detection_folder}")
        except Exception as e:
            print(f"Error deleting folder {ai_detection_folder}: {e}")

    output_path = os.path.join("temp", "ai_detected.png")
    try:
        subprocess.run([
            "python3", os.path.join("yolov9-main","detect.py"),
            "--weights", "best.pt",
            "--source", image_path,
            "--project", "temp",
            "--name", "ai_detection",
            "--conf", "0.25"
        ], check=True)
        # Assuming the output is saved in the temp/ai_detection directory
        detected_image = os.path.join("temp", "ai_detection", os.path.basename(image_path))
        if os.path.exists(detected_image):
            os.rename(detected_image, output_path)
            return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error running YOLOv9: {e}")
    return None

def generate_images():
    # Clear temp directory first
    if os.path.exists("temp"):
        for file in os.listdir("temp"):
            file_path = os.path.join("temp", file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    number = random.choice(list(number_to_animal.keys()))
    animal = number_to_animal[number]

    with open(os.path.join("data", str(number) + ".json"), "r") as file:
        data = json.load(file)

    image_data = random.choice(data)
    input_image_path = os.path.join("downloaded_images", str(number), image_data["uuid"] + ".png")

    # Load and process the original image with PIL
    pil_image = Image.open(input_image_path)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype("arial.ttf", size=100)
    except IOError:
        font = ImageFont.load_default()

    # Create labeled image with PIL
    labeled_pil = pil_image.copy()
    labeled_draw = ImageDraw.Draw(labeled_pil)

    # Draw all bounding boxes and labels
    for box in image_data["boundingBoxes"]:
        x, y, width, height = box['x'], box['y'], box['width'], box['height']
        label = box['concept']

        # Draw the rectangle
        labeled_draw.rectangle([(x, y), (x + width, y + height)], outline="red", width=2)

        # Add the label
        text_position = (x, max(0, y - 20))  # Position slightly above the rectangle
        # Draw white outline for better visibility
        for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            labeled_draw.text((text_position[0]+dx, text_position[1]+dy), label, fill="white", font=font)
        labeled_draw.text(text_position, label, fill="red", font=font)

    # Create guess image with PIL
    guess_pil = pil_image.copy()
    guess_draw = ImageDraw.Draw(guess_pil)

    # Draw single box for guess image
    filtered_boxes = [box for box in image_data["boundingBoxes"] if box['concept'] == animal]
    if filtered_boxes:
        box_data = random.choice(filtered_boxes)
        x, y, width, height = box_data['x'], box_data['y'], box_data['width'], box_data['height']
        guess_draw.rectangle([(x, y), (x + width, y + height)], outline="red", width=2)

    # Save temporary images
    os.makedirs("temp", exist_ok=True)
    labeled_path = os.path.join("temp", 'labelled.png')
    guess_path = os.path.join("temp", 'guess.png')

    labeled_pil.save(labeled_path)
    guess_pil.save(guess_path)

    return input_image_path, labeled_path, guess_path
def load_and_scale_image(image_path):
    try:
        image = pygame.image.load(image_path)
        image_width = WINDOW_WIDTH * 0.8
        image_height = WINDOW_HEIGHT * 0.7
        scaled_image = pygame.transform.scale(image, (int(image_width), int(image_height)))
        return scaled_image
    except Exception as e:
        print(f"Error loading image: {image_path}, Error: {e}")
        return None

def draw_button(surface, button, text, is_hovered):
    color = DARK_GRAY if is_hovered else GRAY
    pygame.draw.rect(surface, color, button)
    add_text_to_surface(surface, text,
                       (button.centerx - len(text)*7, button.centery - 10),
                       font_size=FONT_SIZE,
                       color=WHITE)

def main():
    global current_image, showing_image
    clock = pygame.time.Clock()

    # Generate initial images
    original_image_path, labeled_image_path, guess_image_path = generate_images()
    ai_image_path = None
    # Create images array with swapped positions for labeled and guess images
    images = [original_image_path, guess_image_path, labeled_image_path, None, None]

    # Extract the animal number from the original image path
    current_animal = None
    if original_image_path:
        path_parts = original_image_path.split(os.sep)
        for part in path_parts:
            if part.isdigit():
                current_animal = number_to_animal[int(part)]
                break

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    showing_image = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, button in enumerate(buttons):
                    if button.collidepoint(mouse_pos):
                        if i == 4:  # Random Image button
                            original_image_path, labeled_image_path, guess_image_path = generate_images()
                            ai_image_path = None
                            # Update current animal
                            path_parts = original_image_path.split(os.sep)
                            for part in path_parts:
                                if part.isdigit():
                                    current_animal = number_to_animal[int(part)]
                                    break
                            # Maintain the same swapped order when generating new images
                            images = [original_image_path, guess_image_path, labeled_image_path, None, None]
                        elif i == 3:  # AI button
                            if original_image_path:
                                ai_image_path = run_yolov9_detection(original_image_path)
                                images[3] = ai_image_path
                                if ai_image_path:
                                    current_image = load_and_scale_image(ai_image_path)
                                    showing_image = True
                        elif images[i] is not None:
                            current_image = load_and_scale_image(images[i])
                            if current_image:
                                showing_image = True
                            if i == 2:
                                print(current_animal)

        screen.fill(WHITE)

        # Draw title
        add_text_to_surface(screen, "Marine Species Viewer",
                           (WINDOW_WIDTH//2 - 150, 20),
                           font_size=TITLE_FONT_SIZE,
                           color=BLACK)

        if showing_image and current_image:
            image_rect = current_image.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 30))
            screen.blit(current_image, image_rect)
            add_text_to_surface(screen, "Press ESC to go back",
                              (WINDOW_WIDTH//2 - 100, WINDOW_HEIGHT - 40),
                              color=BLACK)

        else:
            mouse_pos = pygame.mouse.get_pos()
            for i, button in enumerate(buttons):
                is_hovered = button.collidepoint(mouse_pos)
                draw_button(screen, button, button_labels[i], is_hovered)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
