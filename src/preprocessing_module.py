# src/preprocessing_module.py

import cv2
import numpy as np

def preprocess_for_ocr(image_array):
    """
    Refined preprocessing pipeline based on the original simple version.
    1. Converts to grayscale.
    2. Applies gentle de-skewing ONLY if significant skew is detected.
    3. Applies Gaussian Blur for noise reduction.
    4. Applies adaptive thresholding with potentially adjusted parameters.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # 2. Gentle De-skewing (Hough Transform based)
    try:
        # Invert and threshold for line detection
        inverted = cv2.bitwise_not(gray)
        _, thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength=w//3, maxLineGap=w//10) # Adjusted parameters
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle, ignore vertical lines
                if x2 - x1 != 0:
                    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
                    if -45 < angle < 45: # Consider only near-horizontal lines
                         angles.append(angle)

        # Only rotate if a significant median angle is detected
        if angles:
            median_angle = np.median(angles)
            if abs(median_angle) > 1.0: # Threshold to avoid rotating almost-straight images
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                # Use white background fill
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        
    except Exception as e:
        print(f"Could not perform de-skewing: {e}")
        # If de-skewing fails, just continue with the original grayscale image

    # 3. Gaussian Blur (instead of Median) - sometimes better for text
    # You can experiment between GaussianBlur and medianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # blurred = cv2.medianBlur(gray, 3) # alternative from original

    # 4. Adaptive Thresholding - Parameters might need tuning
    # Block size (e.g., 15, 17) and C value (e.g., 7, 9) can be adjusted
    preprocessed = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, # Increased block size slightly
        7  # Increased C value slightly
    )

    return preprocessed


def preprocess_for_trocr(image_array):
    """
    A gentle preprocessing pipeline specifically for TrOCR.
    1. Converts to grayscale.
    2. Removes horizontal ruled lines.
    DOES NOT binarize, which preserves cursive strokes.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # 2. Remove Horizontal Ruled Lines
    # Invert image (light text on dark background)
    inverted = cv2.bitwise_not(gray)
    
    # Detect horizontal lines (which are now white)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)) # Use a long kernel
    detected_lines = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Subtract the white lines from the inverted image
    no_lines_inverted = cv2.subtract(inverted, detected_lines)
    
    # Invert back to dark text on a light/clean background
    final_image = cv2.bitwise_not(no_lines_inverted)
    
    return final_image

def preprocess_for_graphology(image_array):
    """
    Prepares an image for graphological feature extraction.
    Does not binarize, as grayscale info is needed for pressure.
    Returns:
    - gray: The original grayscale image.
    - thresh_inv: An inverted binary image (white text) for contour detection.
    - lines: A list of cropped image arrays, each containing one line of text.
    """
    # 1. Convert to grayscale and blur
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Create an inverted binary image for contour detection
    # This helps find contours on light backgrounds
    thresh_inv = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 3. Segment the image into lines of text
    # Dilate horizontally to connect words into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    dilated = cv2.dilate(thresh_inv, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from top to bottom
    line_bboxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])
    
    lines = []
    for (x, y, w, h) in line_bboxes:
        if h > 10 and w > 50: # Filter out small noise
            # Crop the original grayscale image
            line_crop = gray[y:y+h, x:x+w]
            lines.append(line_crop)
            
    return gray, thresh_inv, lines