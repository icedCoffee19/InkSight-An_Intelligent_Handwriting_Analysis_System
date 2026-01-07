# src/graphology_module.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_graphological_features(gray_image, binary_image, lines):
    """
    Extracts a comprehensive set of graphological features based on Table 2.1.
    Note: Loops, Shape, and Connectivity are highly complex and omitted.
    """
    features = {
        'pressure': 0, 'letter_size': 0, 'slant': 0,
        'baseline_slope': 0, 'word_spacing': 0, 'line_spacing': 0, 'left_margin': 0
    }

    # Find all contours (letters/words) from the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return features # Return empty if no text found

    # 1. Pen Pressure (Proxy) [cite: 97]
    # Average grayscale value of the "ink" pixels
    ink_pixels = gray_image[binary_image == 255]
    if ink_pixels.any():
        features['pressure'] = np.mean(ink_pixels)

    # 2. Letter Size & Slant (using word bounding boxes)
    word_bboxes = []
    word_angles = []
    # Dilate slightly to get word contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_words = cv2.dilate(binary_image, kernel, iterations=1)
    word_contours, _ = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    heights = []
    for c in word_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10: # Filter noise
            word_bboxes.append((x, y, w, h))
            heights.append(h)
            
            # Slant [cite: 96]
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            if w < h: # Verticalish
                angle = 90 + angle
            if abs(angle) < 45: # Only include non-vertical angles
                word_angles.append(angle)

    if heights:
        features['letter_size'] = np.median(heights) # [cite: 95]
    if word_angles:
        features['slant'] = np.median(word_angles) # [cite: 96]
    
    # Sort word boxes top-to-bottom, then left-to-right
    word_bboxes.sort(key=lambda b: (b[1], b[0]))
    
    # 3. Margins, Spacing, and Baseline
    line_y_coords = []
    word_gaps = []
    left_margins = []
    
    if not word_bboxes:
        return features

    # Group words by line
    lines_of_words = []
    current_line = []
    current_y = word_bboxes[0][1]
    
    for (x, y, w, h) in word_bboxes:
        if abs(y - current_y) > h * 0.7: # New line
            lines_of_words.append(sorted(current_line, key=lambda b: b[0]))
            current_line = [(x, y, w, h)]
            current_y = y
        else:
            current_line.append((x, y, w, h))
    lines_of_words.append(sorted(current_line, key=lambda b: b[0]))

    baseline_slopes = []
    
    for line in lines_of_words:
        if not line:
            continue
            
        # Left Margin [cite: 100]
        left_margins.append(line[0][0])
        
        # Word Spacing [cite: 99]
        for i in range(len(line) - 1):
            bbox1 = line[i]
            bbox2 = line[i+1]
            gap = bbox2[0] - (bbox1[0] + bbox1[2])
            if gap > 0:
                word_gaps.append(gap)
                
        # Baseline [cite: 98]
        # Use the (x, y+h) bottom-center of each word
        points_x = [b[0] + b[2] / 2 for b in line]
        points_y = [b[1] + b[3] for b in line]
        if len(points_x) > 1:
            m, _ = np.polyfit(points_x, points_y, 1) # m = slope
            baseline_slopes.append(m)
            line_y_coords.append(np.median(points_y))

    if word_gaps:
        features['word_spacing'] = np.median(word_gaps)
    if left_margins:
        features['left_margin'] = np.median(left_margins)
    if baseline_slopes:
        features['baseline_slope'] = np.median(baseline_slopes)

    # Line Spacing [cite: 94]
    line_gaps = np.diff(line_y_coords)
    if len(line_gaps) > 0:
        features['line_spacing'] = np.median(line_gaps)
        
    return features


def get_personality_profile(features):
    """
    Infers personality traits based on a rule-based system from Table 2.1.
    Returns both quantitative scores (for spider chart) and descriptions.
    """
    quantitative = {
        'Sociability': 0.5, 'Focus': 0.5, 'Intensity': 0.5,
        'Optimism': 0.5, 'Discipline': 0.5, 'Spontaneity': 0.5
    }
    descriptive = {}

    # 1. Sociability vs. Reserved (from Slant & Word Spacing)
    if features['slant'] > 5: # Right slant [cite: 96]
        quantitative['Sociability'] = 0.8
        descriptive['Outlook'] = "Sociable, expressive, approach-oriented." #[cite: 96]
    elif features['slant'] < -5: # Left slant [cite: 96]
        quantitative['Sociability'] = 0.2
        descriptive['Outlook'] = "Reserved, withdrawn, avoids conflict." #[cite: 96]
    else: # Vertical slant [cite: 96]
        quantitative['Sociability'] = 0.4
        descriptive['Outlook'] = "Controlled, independent, balanced." #[cite: 96]

    if features['word_spacing'] > features['letter_size']: # Wide spacing [cite: 99]
        quantitative['Sociability'] -= 0.1
        descriptive['Social Type'] = "Values space, independent." #[cite: 99]
    elif features['word_spacing'] < features['letter_size'] * 0.5: # Narrow spacing [cite: 99]
        quantitative['Sociability'] += 0.2
        descriptive['Social Type'] = "Sociable, seeks closeness." #[cite: 99]

    # 2. Focus vs. Assertive (from Letter Size)
    if features['letter_size'] > 50: # Large letters [cite: 95] (Threshold is a guess)
        quantitative['Focus'] = 0.3
        descriptive['Focus'] = "Outgoing, assertive, big-picture oriented." #[cite: 95]
    elif features['letter_size'] < 25: # Small letters [cite: 95]
        quantitative['Focus'] = 0.9
        descriptive['Focus'] = "Focused, introverted, detail-oriented." #[cite: 95]
    else:
        descriptive['Focus'] = "Balanced and adaptable."

    # 3. Intensity vs. Sensitivity (from Pressure)
    if features['pressure'] < 120: # Dark ink (0=black, 255=white) [cite: 97]
        quantitative['Intensity'] = 0.8
        descriptive['Intensity'] = "Intense, forceful, high energy." #[cite: 97]
    elif features['pressure'] > 180: # Light ink [cite: 97]
        quantitative['Intensity'] = 0.2
        descriptive['Intensity'] = "Sensitive, empathetic, low arousal." #[cite: 97]
    else:
        descriptive['Intensity'] = "Balanced emotional intensity."

    # 4. Optimism vs. Pessimism (from Baseline)
    if features['baseline_slope'] < -0.05: # Falling slope [cite: 98]
        quantitative['Optimism'] = 0.2
        descriptive['Mood'] = "Pessimistic, tired, or discouraged." #[cite: 98]
    elif features['baseline_slope'] > 0.05: # Rising slope [cite: 98]
        quantitative['Optimism'] = 0.8
        descriptive['Mood'] = "Optimistic, energetic, ambitious." #[cite: 98]
    else:
        quantitative['Optimism'] = 0.5
        descriptive['Mood'] = "Steady, balanced, and stable." #[cite: 98]

    # 5. Discipline (from Baseline & Slant)
    if -0.05 <= features['baseline_slope'] <= 0.05: # Straight baseline [cite: 98]
        quantitative['Discipline'] = 0.8
    else:
        quantitative['Discipline'] = 0.3 # Wavy baseline

    if -5 <= features['slant'] <= 5: # Vertical slant [cite: 96]
        quantitative['Discipline'] += 0.2 # controlled
    
    # 6. Spontaneity (from Left Margin)
    if features['left_margin'] > 40: # Wide margin [cite: 100]
        quantitative['Spontaneity'] = 0.2
        descriptive['Planning'] = "Cautious, planned, looks to the future." #[cite: 100]
    elif features['left_margin'] < 20: # Narrow margin [cite: 100]
        quantitative['Spontaneity'] = 0.9
        descriptive['Planning'] = "Spontaneous, impulsive, lives in the moment." #[cite: 100]

    # Ensure all scores are capped between 0.1 and 1.0
    for k in quantitative:
        quantitative[k] = np.clip(quantitative[k], 0.1, 1.0)
        
    return quantitative, descriptive


def create_spider_chart(quantitative_profile):
    """
    Creates a spider chart (radar chart) from the quantitative trait profile.
    """
    labels = list(quantitative_profile.keys())
    stats = list(quantitative_profile.values())
    
    # Calculate angles
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop for the spider chart
    stats_loop = stats + stats[:1]
    angles_loop = angles + angles[:1]
    
    # --- CHANGE 1: Figure size is now (4, 4) ---
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    
    ax.fill(angles_loop, stats_loop, color='blue', alpha=0.25)
    ax.plot(angles_loop, stats_loop, color='blue', linewidth=2)
    
    # Set radial labels (y-ticks)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Low", "Mid", "High", "V. High"])
    
    # --- CHANGE 2: Adjust ylim to create padding ---
    ax.set_ylim(0, 1.2)
    
    # Set the angle for the radial labels to avoid overlap
    ax.set_rlabel_position(22.5) 
    
    # Set category labels (x-ticks)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    
    # --- NEW: Manually adjust label alignment ---
    # We will adjust the alignment of each label to push it "outward"
    # from the plot center, preventing overlap.
    
    # Get all the xtick label objects
    xtick_labels = ax.get_xticklabels()
    
    # 'Sociability' is at 0 degrees (index 0) -> align left
    xtick_labels[0].set_horizontalalignment('left')
    
    # 'Focus' (index 1) and 'Intensity' (index 2) are at the top
    xtick_labels[1].set_verticalalignment('bottom')
    xtick_labels[2].set_verticalalignment('bottom')
    
    # 'Optimism' is at 180 degrees (index 3) -> align right
    xtick_labels[3].set_horizontalalignment('right')

    # 'Discipline' (index 4) and 'Spontaneity' (index 5) are at the bottom
    xtick_labels[4].set_verticalalignment('top')
    xtick_labels[5].set_verticalalignment('top')
    
    return fig