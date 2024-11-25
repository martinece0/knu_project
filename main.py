import cv2
import numpy as np

def four_point_transform(image, pts):
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

def remove_blue_areas(plate_image):
    # Convertir l'image en HSV pour mieux détecter la couleur bleue
    hsv = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

    # Définir les plages pour détecter le bleu
    lower_blue = np.array([90, 50, 50])  # Teinte bleue minimale
    upper_blue = np.array([130, 255, 255])  # Teinte bleue maximale

    # Masque pour détecter les zones bleues
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Trouver les contours des zones bleues
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Récupérer les limites des zones bleues
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Vérifier si le rectangle détecté est une partie latérale bleue
        if h > plate_image.shape[0] * 0.5 and w < plate_image.shape[1] * 0.3:
            # Supprimer cette zone en la rendant blanche
            plate_image[:, x:x + w] = [255, 255, 255]

    return plate_image

# Charger l'image principale
image_path = "images/voiture_9.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Erreur : Impossible de charger l'image.")
else:
    height, width = image.shape[:2]
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
        if len(approx) == 4:  # Contours rectangulaires
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
        # Extraire les coins du rectangle détecté
        pts = largest_contour.reshape(4, 2)
        warped_plate = four_point_transform(image_resized, pts)

        # Enlever les zones bleues
        cleaned_plate = remove_blue_areas(warped_plate)

        # Afficher la plaque corrigée sans bleu
        cv2.imshow("Plaque nettoyée", cleaned_plate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Aucune plaque détectée.")
