import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt

# Define class names
class_names = ['ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 
               'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya']

# Function to display image
def show_image(image, title=''):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to mask image
def mask_image(image):
    image_np = np.array(image)
    hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 135])
    mask = cv2.inRange(hsv_image, lower_black, upper_black)
    binary_image = cv2.bitwise_not(mask)
    return Image.fromarray(binary_image)  # Convert to PIL image

# Function to check if a character image is valid based on the proportion of black pixels
def is_valid_character(char_image):
    char_image_np = np.array(char_image)  # Convert PIL image to numpy array
    total_pixels = char_image_np.size
    black_pixels = np.sum(char_image_np == 0)
    black_ratio = black_pixels / total_pixels
    return 0.05 <= black_ratio <= 0.70

# Function for image preprocessing and character segmentation
def preprocess_and_segment(image):
    # Convert PIL image to numpy array (RGB)
    image_np = np.array(image.convert('RGB'))
    # Convert image from RGB to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Thresholding to get binary image
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Use morphological operations to thin characters
    kernel = np.ones((1, 1), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    # Find contours in binary image
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_images = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_image = eroded_image[y:y+h, x:x+w]
        char_image_negated = cv2.bitwise_not(char_image)
        border_size = 10
        char_image_with_border = cv2.copyMakeBorder(
            char_image_negated, 
            border_size, border_size, border_size, border_size, 
            cv2.BORDER_CONSTANT, 
            value=255
        )
        if is_valid_character(char_image_with_border):
            char_images.append((char_image_with_border, x))
    char_images = sorted(char_images, key=lambda x: x[1])
    return char_images, contours

# Function to detect spaces between characters based on Q3
def detect_spaces(contours, valid_chars):
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    spaces = []
    positions = []
    space_widths = []

    valid_contours = [c for c in contours if cv2.boundingRect(c)[0] in [x[1] for x in valid_chars]]

    for i in range(1, len(valid_contours)):
        x_prev, _, w_prev, _ = cv2.boundingRect(valid_contours[i - 1])
        x_curr, _, _, _ = cv2.boundingRect(valid_contours[i])
        space_width = x_curr - (x_prev + w_prev)
        space_widths.append(space_width)

    # Calculate Q3 (third quartile)
    if space_widths:
        q3 = np.percentile(space_widths, 75)
        for i in range(len(space_widths)):
            if space_widths[i] > q3:
                if i + 1 < len(valid_contours):  # Ensure the next index is within range
                    x_prev, _, w_prev, _ = cv2.boundingRect(valid_contours[i])
                    x_curr, _, _, _ = cv2.boundingRect(valid_contours[i + 1])
                    positions.append((x_prev + w_prev, x_curr))

    return positions

# Function to count characters left of spaces
def count_chars_left_of_spaces(positions, valid_chars):
    counts = []
    for (x1, _) in positions:
        count = sum(1 for char in valid_chars if char[1] < x1)
        counts.append(count)
    return counts

# Function to add spaces to characters
def add_spaces_to_chars(segmented_chars, positions, char_counts_left_of_spaces):
    result = []
    char_index = 0
    pos_index = 0

    for i, (char_image, x) in enumerate(segmented_chars):
        if pos_index < len(positions) and x >= positions[pos_index][0]:
            # Insert space if current character is at or past the position where space should be inserted
            result.append(' ')
            pos_index += 1

        result.append((char_image, x))  # Add the character to the result list

    return result

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Define the transformations
def transform(image):
    image = image.resize((128, 128))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to predict the class
def predict(image, model):
    image = transform(image)
    outputs = model.predict(image)
    predicted = np.argmax(outputs, axis=1)
    return class_names[predicted[0]]

# Streamlit app
st.title("Aksara Jawa Detection")

# Camera input with custom CSS applied
image_data = st.camera_input("Take a picture", key="camera_input")

if image_data is not None:
    image = Image.open(io.BytesIO(image_data.getvalue()))
    # Apply masking
    masked_image = mask_image(image)
    # Display the masked image
    st.image(masked_image, caption='Masked Image', use_column_width=True)
    # Segment characters from the masked image
    segmented_chars, contours = preprocess_and_segment(masked_image)
    # Filter valid characters
    valid_chars = [(char_image, x) for char_image, x in segmented_chars if is_valid_character(char_image)]
    # Detect spaces
    positions = detect_spaces(contours, valid_chars)
    # Count characters left of each space
    char_counts_left_of_spaces = count_chars_left_of_spaces(positions, valid_chars)
    # Add spaces to characters
    segmented_chars_with_spaces = add_spaces_to_chars(valid_chars, positions, char_counts_left_of_spaces)
    if segmented_chars_with_spaces:
        # Display the segmented characters and predictions
        st.write("Segmented Characters and Predictions:")
        text_output = []
        for i, item in enumerate(segmented_chars_with_spaces):
            if isinstance(item, str):  # Check if item is a space
                text_output.append(item)
            else:
                char_image, x = item
                char_image_pil = Image.fromarray(char_image)
                char_class = predict(char_image_pil, model)
                text_output.append(char_class)  # Add predicted class to text_output
                st.image(char_image, caption=f'Character {i}: {char_class}', use_column_width=True)
        
        # Display detected spaces and final output text
        st.write("Detected Spaces:")
        for (x1, x2) in positions:
            st.write(f"Space between {x1} and {x2}")

        st.write("Final Output Text:")
        final_text_output = "".join(text_output)  # Join text_output list into a single string
        st.write(final_text_output)  # Display final text output with spaces
    else:
        st.write("No characters detected.")
    
    # Visualize detected spaces
    image_np = np.array(masked_image)
    for (x1, x2) in positions:
        cv2.rectangle(image_np, (x1, 0), (x2, image_np.shape[0]), (0, 255, 0), 2)
    
    st.image(image_np, caption='Detected Spaces', use_column_width=True)
