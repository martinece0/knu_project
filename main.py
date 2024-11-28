import cv2
import numpy as np
import pytesseract
import re

# Function to apply a four-point perspective transform
def four_point_transform(image, pts):
    """
    Applies a perspective transformation to focus on a quadrilateral region.

    Args:
        image (ndarray): Original image.
        pts (ndarray): Four points defining the region.

    Returns:
        ndarray: Warped image containing the transformed region.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# Function to remove blue areas (e.g., EU stars) from the license plate
def remove_blue_areas(plate_image):
    """
    Detects and removes blue areas from the license plate.

    Args:
        plate_image (ndarray): License plate image.

    Returns:
        ndarray: Image with blue areas replaced by white.
    """
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > plate_image.shape[0] * 0.5 and w < plate_image.shape[1] * 0.3:
            plate_image[:, x:x + w] = [255, 255, 255]

    return plate_image

# Function to enhance the contrast of the license plate text
def enhance_plate_contrast(plate_image):
    """
    Converts the license plate image to grayscale and enhances its contrast.

    Args:
        plate_image (ndarray): License plate image.

    Returns:
        ndarray: Processed image with improved contrast.
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    alpha = 2  # Contrast level
    beta = 0   # Brightness level
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    _, thresholded = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

# Function to correct OCR errors for French license plates
def correct_plate_text(text):
    """
    Corrects common OCR errors in license plate text.

    Args:
        text (str): Raw OCR-detected text.

    Returns:
        str: Corrected and formatted license plate text.
    """
    corrected_text = ""
    for i, char in enumerate(text):
        if 2 <= i <= 4:
            char = {'G': '6', 'I': '1', 'B': '8'}.get(char, char)
        else:
            char = {'0': 'O', 'O': 'D', '6': 'G', '8': 'B'}.get(char, char)
        corrected_text += char

    corrected_text = add_missing_hyphens(corrected_text)
    return corrected_text

# Function to add missing hyphens to license plate text
def add_missing_hyphens(text):
    """
    Adds hyphens to the license plate text based on its length.

    Args:
        text (str): Text without hyphens.

    Returns:
        str: Formatted text with hyphens.
    """
    if len(text) == 8 and text[2].isdigit():
        return text[:2] + '-' + text[2:5] + '-' + text[5:]
    elif len(text) == 7 and text[1].isdigit():
        return text[:1] + '-' + text[1:4] + '-' + text[4:]
    return text

# Function to validate license plate format
def is_valid_plate(text):
    """
    Validates the text against the French license plate format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if valid, False otherwise.
    """
    plate_pattern = r'^([A-Z]{2}-\d{3}-[A-Z]{2}|[A-Z]-\d{3}-[A-Z]{2})$'
    return re.match(plate_pattern, text) is not None

# Main script
if __name__ == "__main__":
    image_path = "images/voiture_28.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image.")
    else:
        height, width = image.shape[:2]
        scale_factor = 2 if width > 1000 else 1.2
        image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

        cv2.imshow("Original Image", image_resized)

        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        largest_area = 0
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                area = cv2.contourArea(contour)
                rect_area = w * h
                extent = area / rect_area

                if 1.5 < aspect_ratio < 6 and 0.3 < extent < 1.2 and area > 1000:
                    if area > largest_area:
                        largest_area = area
                        largest_contour = approx

        if largest_contour is not None:
            pts = largest_contour.reshape(4, 2)
            warped_plate = four_point_transform(image_resized, pts)

            cleaned_plate = remove_blue_areas(warped_plate)
            enhanced_plate = enhance_plate_contrast(cleaned_plate)

            # Draw a green contour around the plate
            cv2.polylines(image_resized, [largest_contour], isClosed=True, color=(0, 255, 0), thickness=3)

            cv2.imshow("Plate with Green Contour", image_resized)
            cv2.imshow("Cleaned Plate", cleaned_plate)
            cv2.imshow("Enhanced Plate", enhanced_plate)

            custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNPQRSTVWXYZ0123456789'
            plate_text = pytesseract.image_to_string(enhanced_plate, config=custom_config)

            plate_text = correct_plate_text(plate_text)

            # Display results
            print("\n" + "=" * 50)
            print(" 🚗 License Plate Detection Result 🚗 ")
            print("=" * 50)

            print(f"\nDetected Text: {plate_text.strip()}")

            if is_valid_plate(plate_text.strip()):
                print("\n✅ Valid Plate Detected!")
                print(f"   Plate Number: {plate_text.strip()}")
            else:
                print("\n❌ Invalid Plate Detected.")
                print(f"   Raw Detected Text: {plate_text.strip()}")

            print("\n" + "=" * 50)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
