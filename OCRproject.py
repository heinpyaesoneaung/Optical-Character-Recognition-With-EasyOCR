import easyocr
import cv2
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

start_time = time.time()

# ----- User Inputs -----
image_path = "thai_text.gif"   # your input image
output_text_file = ""
translated_image_path = ""

# ----- Read Image -----
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not open image: {image_path}")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(image_pil)

# ----- Initialize EasyOCR Reader -----
reader = easyocr.Reader(['th','en'])

# ----- Detect Text -----
results = reader.readtext(image)

# ----- Print, Translate, Save text, Overlay -----
print("\n===== OCR + Translation =====")
with open(output_text_file, "w") as f:
    for (bbox, text, prob) in results:
        # Step 1: Detect and translate
        text_en = GoogleTranslator(source='auto', target='en').translate(text)

        # Console output
        print(f"Original: {text} | English: {text_en} (conf: {prob:.2f})")
        f.write(f"{text_en}")

        # Overlay translated text on image
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x1, y1 = map(int, top_left)
        x2, y2 = map(int, bottom_right)

        # Erase original text
        draw.rectangle([x1, y1, x2, y2], fill="white")


        # Dynamic font size
        box_height = y2 - y1
        font_size = max(16, box_height // 2)
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        draw.multiline_text((x1, y1), text_en, fill="black", font=font)

# Save translated image
image_pil.save(translated_image_path)
print(f"\n Translated image saved to: {translated_image_path}")
print(f"   Translated text saved to: {output_text_file}")

end_time = time.time()
print(f"\n Total execution time: {end_time - start_time:.2f} seconds")

# ----- Display original and translated images -----
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    cv2.rectangle(image_rgb, top_left, bottom_right, (255, 0, 0), 2)

plt.figure("OCR using EasyOCR",figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(np.array(image_pil))
plt.title("Translated Image")
plt.axis("off")

plt.show()
