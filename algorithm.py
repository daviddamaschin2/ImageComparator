import numpy as np
import cv2
from typing import Tuple


class ChangeDetector:
    """Detect changes between two photos taken at different times."""

    def __init__(self, threshold=25, min_contour_area=100):
        """
        Args:
            threshold: Sensitivity for change detection (0-255, lower=more sensitive)
            min_contour_area: Minimum pixel area to consider as a change
        """
        self.threshold = threshold
        self.min_contour_area = min_contour_area

    def align_images(self, img1, img2):
        """Align images to compensate for camera movement."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        orb = cv2.ORB_create(5000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # Match features
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        if len(matches) < 10:
            # Not enough matches, return original
            return img2

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:500]])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:500]])

        # Find homography
        h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        # Warp img2 to align with img1
        height, width = img1.shape[:2]
        aligned = cv2.warpPerspective(img2, h, (width, height))

        return aligned

    def detect_changes(self, img1_path, img2_path, align=True):
        """
        Detect changes between two images.

        Returns:
            dict with 'visualization', 'mask', 'changes', 'stats'
        """
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise ValueError("Could not load images")

        # Resize img2 to match img1
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Align images if requested
        if align:
            img2_aligned = self.align_images(img1, img2)
        else:
            img2_aligned = img2

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)

        # Threshold the difference
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of changes
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        significant_contours = [c for c in contours
                                if cv2.contourArea(c) > self.min_contour_area]

        # Create visualizations
        visualization = img1.copy()
        mask = np.zeros_like(gray1)
        changes_only = np.zeros_like(img1)

        for contour in significant_contours:
            # Draw bounding box on visualization
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Fill mask
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Extract changed region
            cv2.drawContours(changes_only, [contour], -1, (255, 255, 255), -1)

        # Apply mask to show only changes
        changes_only = cv2.bitwise_and(img2_aligned, changes_only)

        # Calculate statistics
        total_pixels = img1.shape[0] * img1.shape[1]
        changed_pixels = np.sum(mask > 0)
        change_percentage = (changed_pixels / total_pixels) * 100

        stats = {
            'total_changes': len(significant_contours),
            'changed_pixels': changed_pixels,
            'total_pixels': total_pixels,
            'change_percentage': change_percentage
        }

        return {
            'visualization': visualization,  # Original with boxes
            'mask': mask,  # Binary mask of changes
            'changes_only': changes_only,  # Only changed regions
            'diff_image': diff,  # Raw difference
            'threshold_image': thresh,  # Binary threshold
            'stats': stats
        }


# Example usage
if __name__ == "__main__":
    detector = ChangeDetector(threshold=25, min_contour_area=100)

    result = detector.detect_changes('Capture.jpeg', 'Capture2.jpeg', align=True)

    # Save results
    cv2.imwrite('changes_detected.jpg', result['visualization'])
    cv2.imwrite('changes_mask.jpg', result['mask'])
    cv2.imwrite('changes_only.jpg', result['changes_only'])

    # Print statistics
    print(f"Changes detected: {result['stats']['total_changes']}")
    print(f"Changed area: {result['stats']['change_percentage']:.2f}%")
    print(f"Changed pixels: {result['stats']['changed_pixels']:,}")