import cv2
import numpy as np

image_path = "images/voiture_15.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Erreur : Impossible de charger l'image.")
else:
    height, width = image.shape[:2]
    # Ajuste le facteur en fonction de la largeur de l'image
    scale_factor = 2 if width > 1000 else 1.2
    image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

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

            if 1.5 < aspect_ratio < 6 and 0.3 < extent < 1.2 and area > 500:
                if area > largest_area:
                    largest_area = area
                    largest_contour = approx

    if largest_contour is not None:
        cv2.drawContours(image_resized, [largest_contour], -1, (0, 255, 0), 3)

    cv2.imshow("Detection de plaque", image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()