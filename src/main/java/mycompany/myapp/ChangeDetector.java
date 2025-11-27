package mycompany.myapp;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_features2d.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;

public class ChangeDetector {

    private int threshold;
    private int minContourArea;
    private boolean enableAlignment;

    public ChangeDetector(int threshold, int minContourArea, boolean enableAlignment) {
        this.threshold = threshold;
        this.minContourArea = minContourArea;
        this.enableAlignment = enableAlignment;
    }

    /**
     * Align img2 to match img1 using feature-based homography
     */
    private BufferedImage alignImages(BufferedImage img1, BufferedImage img2) {
        // Convert BufferedImages to OpenCV Mat
        Mat mat1 = bufferedImageToMat(img1);
        Mat mat2 = bufferedImageToMat(img2);

        // Convert to grayscale
        Mat gray1 = new Mat();
        Mat gray2 = new Mat();
        cvtColor(mat1, gray1, COLOR_BGR2GRAY);
        cvtColor(mat2, gray2, COLOR_BGR2GRAY);

        // Detect ORB features
        ORB orb = ORB.create();
        orb.setMaxFeatures(5000);
        KeyPointVector keypoints1 = new KeyPointVector();
        KeyPointVector keypoints2 = new KeyPointVector();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();

        orb.detectAndCompute(gray1, new Mat(), keypoints1, descriptors1);
        orb.detectAndCompute(gray2, new Mat(), keypoints2, descriptors2);

        // Match features using BFMatcher
        BFMatcher matcher = BFMatcher.create(NORM_HAMMING, true);
        DMatchVector matches = new DMatchVector();
        matcher.match(descriptors1, descriptors2, matches);

        // Check if we have enough matches
        if (matches.size() < 10) {
            System.out.println("Not enough matches found. Returning original image.");
            mat1.release();
            mat2.release();
            gray1.release();
            gray2.release();
            return img2;
        }

        // Sort matches by distance
        List<DMatch> matchList = new ArrayList<>();
        for (int i = 0; i < matches.size(); i++) {
            matchList.add(matches.get(i));
        }
        matchList.sort((a, b) -> Float.compare(a.distance(), b.distance()));

        // Keep only best matches (top 500 or all if fewer)
        int numGoodMatches = Math.min(500, matchList.size());

        // Extract matched keypoints
        Point2fVector points1 = new Point2fVector();
        Point2fVector points2 = new Point2fVector();

        for (int i = 0; i < numGoodMatches; i++) {
            DMatch match = matchList.get(i);
            points1.push_back(new Point2f(
                    keypoints1.get(match.queryIdx()).pt().x(),
                    keypoints1.get(match.queryIdx()).pt().y()
            ));
            points2.push_back(new Point2f(
                    keypoints2.get(match.trainIdx()).pt().x(),
                    keypoints2.get(match.trainIdx()).pt().y()
            ));
        }

        // Ensure enough raw matches
        if (numGoodMatches < 4) {
            System.out.println("Not enough matches for homography.");
            return img2;
        }

        // Build point mats explicitly (Nx1, CV_32FC2)
        Mat points1Mat = new Mat(numGoodMatches, 1, CV_32FC2);
        Mat points2Mat = new Mat(numGoodMatches, 1, CV_32FC2);
        FloatPointer p1 = new FloatPointer(points1Mat.data());
        FloatPointer p2 = new FloatPointer(points2Mat.data());
        for (int i = 0; i < numGoodMatches; i++) {
            Point2f pt1 = points1.get(i);
            Point2f pt2 = points2.get(i);
            p1.put(i * 2,     pt1.x());
            p1.put(i * 2 + 1, pt1.y());
            p2.put(i * 2,     pt2.x());
            p2.put(i * 2 + 1, pt2.y());
        }

        Mat mask = new Mat(); // will be filled by RANSAC
        Mat homography = findHomography(points2Mat, points1Mat, RANSAC, 5.0, mask, 2000, 0.995);
        if (homography == null || homography.empty()) {
            System.out.println("Homography computation failed.");
            points1Mat.release();
            points2Mat.release();
            return img2;
        }

        // Optional: refine only if enough inliers
        int inliers = countNonZero(mask);
        if (inliers < 4) {
            System.out.println("Too few inliers after RANSAC.");
            points1Mat.release();
            points2Mat.release();
            homography.release();
            return img2;
        }

        // Warp
        Mat aligned = new Mat();
        warpPerspective(mat2, aligned, homography, new Size(mat1.cols(), mat1.rows()));

        BufferedImage result = matToBufferedImage(aligned);

        // Clean up all Mats
        mat1.release();
        mat2.release();
        gray1.release();
        gray2.release();
        aligned.release();

        return result;
    }

    /**
     * Convert BufferedImage to OpenCV Mat
     */
    private Mat bufferedImageToMat(BufferedImage img) {
        // Convert to BGR format that OpenCV expects
        int width = img.getWidth();
        int height = img.getHeight();

        Mat mat = new Mat(height, width, CV_8UC3);

        byte[] data = new byte[width * height * 3];
        int[] pixels = img.getRGB(0, 0, width, height, null, 0, width);

        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            data[i * 3] = (byte) ((pixel) & 0xFF);        // Blue
            data[i * 3 + 1] = (byte) ((pixel >> 8) & 0xFF);  // Green
            data[i * 3 + 2] = (byte) ((pixel >> 16) & 0xFF); // Red
        }

        mat.data().put(data);
        return mat;
    }

    /**
     * Convert OpenCV Mat to BufferedImage
     */
    private BufferedImage matToBufferedImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        int width = mat.cols();
        int height = mat.rows();
        int channels = mat.channels();

        if (channels == 3) {
            BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            byte[] data = new byte[width * height * 3];
            mat.data().get(data);

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int index = (y * width + x) * 3;
                    int b = data[index] & 0xFF;
                    int g = data[index + 1] & 0xFF;
                    int r = data[index + 2] & 0xFF;
                    img.setRGB(x, y, (r << 16) | (g << 8) | b);
                }
            }
            return img;
        }

        return null;
    }

    /**
     * Main method to detect changes between two images
     */
    public ChangeResult detectChanges(String img1Path, String img2Path) throws IOException {
        // Load images
        BufferedImage img1 = ImageIO.read(new File(img1Path));
        BufferedImage img2 = ImageIO.read(new File(img2Path));

        // Resize img2 to match img1 if needed
        if (img1.getWidth() != img2.getWidth() || img1.getHeight() != img2.getHeight()) {
            img2 = resizeImage(img2, img1.getWidth(), img1.getHeight());
        }

        // Convert to grayscale
        int[][] gray1 = toGrayscale(img1);
        int[][] gray2 = toGrayscale(img2);

        // Apply Gaussian blur
        gray1 = gaussianBlur(gray1, 5);
        gray2 = gaussianBlur(gray2, 5);

        // Calculate absolute difference
        int[][] diff = absoluteDifference(gray1, gray2);

        diff = gaussianBlur(diff);
        boolean[][] binaryMask = applyThreshold(diff, threshold);

        binaryMask = fillHoles(binaryMask);

        // Find contours
        List<Contour> contours = findContours(binaryMask);

        // Filter by area
        List<Contour> significantContours = new ArrayList<>();
        for (Contour c : contours) {
            if (c.area >= minContourArea) {
                significantContours.add(c);
            }
        }

        // Create visualizations
        BufferedImage visualization = drawContours(img1, significantContours);
        BufferedImage changesOnly = extractChanges(img2, binaryMask);

        // Calculate statistics
        int changedPixels = countTruePixels(binaryMask);
        int totalPixels = img1.getWidth() * img1.getHeight();
        double changePercentage = (changedPixels * 100.0) / totalPixels;

        return new ChangeResult(
                visualization,
                changesOnly,
                binaryMask,
                significantContours.size(),
                changedPixels,
                totalPixels,
                changePercentage
        );
    }

    /**
     * Convert RGB image to grayscale
     */
    private int[][] toGrayscale(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        int[][] gray = new int[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Color c = new Color(img.getRGB(x, y));
                int grayValue = (int) (0.299 * c.getRed() + 0.587 * c.getGreen() + 0.114 * c.getBlue());
                gray[y][x] = grayValue;
            }
        }
        return gray;
    }


    private boolean[][] fillHoles(boolean[][] mask) {
        int height = mask.length;
        int width = mask[0].length;
        boolean[][] result = new boolean[height][width];

        // Copy original mask
        for (int y = 0; y < height; y++) {
            System.arraycopy(mask[y], 0, result[y], 0, width);
        }

        // Flood fill from edges to find background
        boolean[][] background = new boolean[height][width];
        java.util.Deque<int[]> queue = new java.util.ArrayDeque<>();

        // Add all edge pixels that are false (background)
        for (int x = 0; x < width; x++) {
            if (!mask[0][x]) queue.offer(new int[]{x, 0});
            if (!mask[height-1][x]) queue.offer(new int[]{x, height-1});
        }
        for (int y = 0; y < height; y++) {
            if (!mask[y][0]) queue.offer(new int[]{0, y});
            if (!mask[y][width-1]) queue.offer(new int[]{width-1, y});
        }

        // Flood fill background
        while (!queue.isEmpty()) {
            int[] p = queue.poll();
            int x = p[0], y = p[1];

            if (!inBounds(x, y, width, height) || background[y][x] || mask[y][x]) {
                continue;
            }

            background[y][x] = true;
            queue.offer(new int[]{x+1, y});
            queue.offer(new int[]{x-1, y});
            queue.offer(new int[]{x, y+1});
            queue.offer(new int[]{x, y-1});
        }

        // Everything not background is foreground (fills holes)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[y][x] = !background[y][x];
            }
        }

        return result;
    }

    /**
     * Apply Gaussian blur for noise reduction
     */
    private int[][] gaussianBlur(int[][] image, int kernelSize) {
        int height = image.length;
        int width = image[0].length;
        int[][] result = new int[height][width];

        // Simple 5x5 Gaussian kernel
        double[][] kernel = {
                {1, 4, 7, 4, 1},
                {4, 16, 26, 16, 4},
                {7, 26, 41, 26, 7},
                {4, 16, 26, 16, 4},
                {1, 4, 7, 4, 1}
        };
        double kernelSum = 273.0;

        int offset = kernelSize / 2;

        for (int y = offset; y < height - offset; y++) {
            for (int x = offset; x < width - offset; x++) {
                double sum = 0;
                for (int ky = 0; ky < kernelSize; ky++) {
                    for (int kx = 0; kx < kernelSize; kx++) {
                        sum += image[y + ky - offset][x + kx - offset] * kernel[ky][kx];
                    }
                }
                result[y][x] = (int) (sum / kernelSum);
            }
        }
        return result;
    }

    private int[][] gaussianBlur(int[][] diff) {
        int height = diff.length;
        int width = diff[0].length;
        int[][] blurred = new int[height][width];
        int[][] kernel = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
        int kernelSum = 16;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int sum = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += diff[y + ky][x + kx] * kernel[ky + 1][kx + 1];
                    }
                }
                blurred[y][x] = sum / kernelSum;
            }
        }
        return blurred;
    }

    /**
     * Calculate absolute difference between two grayscale images
     */
    private int[][] absoluteDifference(int[][] img1, int[][] img2) {
        int height = img1.length;
        int width = img1[0].length;
        int[][] diff = new int[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                diff[y][x] = Math.abs(img1[y][x] - img2[y][x]);
            }
        }
        return diff;
    }

    /**
     * Apply binary threshold
     */
    private boolean[][] applyThreshold(int[][] image, int threshold) {
        int height = image.length;
        int width = image[0].length;
        boolean[][] result = new boolean[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                result[y][x] = image[y][x] > threshold;
            }
        }
        return result;
    }

    /**
     * Morphological closing (dilation followed by erosion)
     */
    private boolean[][] morphologicalClose(boolean[][] image, int kernelSize) {
        return erode(dilate(image, kernelSize), kernelSize);
    }

    /**
     * Morphological opening (erosion followed by dilation)
     */
    private boolean[][] morphologicalOpen(boolean[][] image, int kernelSize) {
        return dilate(erode(image, kernelSize), kernelSize);
    }

    /**
     * Dilation operation
     */
    private boolean[][] dilate(boolean[][] image, int kernelSize) {
        int height = image.length;
        int width = image[0].length;
        boolean[][] result = new boolean[height][width];
        int offset = kernelSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean hasWhite = false;
                for (int ky = -offset; ky <= offset; ky++) {
                    for (int kx = -offset; kx <= offset; kx++) {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            if (image[ny][nx]) {
                                hasWhite = true;
                                break;
                            }
                        }
                    }
                    if (hasWhite) break;
                }
                result[y][x] = hasWhite;
            }
        }
        return result;
    }

    /**
     * Erosion operation
     */
    private boolean[][] erode(boolean[][] image, int kernelSize) {
        int height = image.length;
        int width = image[0].length;
        boolean[][] result = new boolean[height][width];
        int offset = kernelSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                boolean allWhite = true;
                for (int ky = -offset; ky <= offset; ky++) {
                    for (int kx = -offset; kx <= offset; kx++) {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            if (!image[ny][nx]) {
                                allWhite = false;
                                break;
                            }
                        } else {
                            allWhite = false;
                            break;
                        }
                    }
                    if (!allWhite) break;
                }
                result[y][x] = allWhite;
            }
        }
        return result;
    }

    /**
     * Find contours using flood fill algorithm
     */
    private List<Contour> findContours(boolean[][] mask) {
        int height = mask.length;
        int width = mask[0].length;
        boolean[][] visited = new boolean[height][width];
        List<Contour> contours = new ArrayList<>();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x] && !visited[y][x]) {
                    Contour contour = new Contour();
                    floodFill(mask, visited, x, y, contour);
                    contours.add(contour);
                }
            }
        }
        return contours;
    }

    /**
     * Flood fill helper for contour detection
     */
    private void floodFill(boolean[][] mask, boolean[][] visited, int startX, int startY, Contour contour) {
        if (!inBounds(startX, startY, mask[0].length, mask.length)
                || visited[startY][startX]
                || !mask[startY][startX]) {
            return;
        }

        java.util.Deque<int[]> stack = new java.util.ArrayDeque<>();
        stack.push(new int[]{startX, startY});

        while (!stack.isEmpty()) {
            int[] p = stack.pop();
            int x = p[0], y = p[1];

            if (!inBounds(x, y, mask[0].length, mask.length)
                    || visited[y][x]
                    || !mask[y][x]) {
                continue;
            }

            visited[y][x] = true;
            contour.points.add(new Point(x, y));
            contour.area++;

            // Update bounding box
            contour.minX = Math.min(contour.minX, x);
            contour.maxX = Math.max(contour.maxX, x);
            contour.minY = Math.min(contour.minY, y);
            contour.maxY = Math.max(contour.maxY, y);

            // 8-way connectivity (includes diagonals)
            stack.push(new int[]{x + 1, y});
            stack.push(new int[]{x - 1, y});
            stack.push(new int[]{x, y + 1});
            stack.push(new int[]{x, y - 1});
            stack.push(new int[]{x + 1, y + 1});  // diagonal
            stack.push(new int[]{x - 1, y - 1});  // diagonal
            stack.push(new int[]{x + 1, y - 1});  // diagonal
            stack.push(new int[]{x - 1, y + 1});  // diagonal
        }
    }

    private boolean inBounds(int x, int y, int width, int height) {
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    /**
     * Draw bounding boxes around contours
     */
    private BufferedImage drawContours(BufferedImage original, List<Contour> contours) {
        BufferedImage result = copyImage(original);
        Graphics2D g = result.createGraphics();
        g.setColor(Color.GREEN);
        g.setStroke(new BasicStroke(2));

        for (Contour c : contours) {
            int width = c.maxX - c.minX;
            int height = c.maxY - c.minY;
            g.drawRect(c.minX, c.minY, width, height);
        }
        g.dispose();
        return result;
    }

    /**
     * Extract only changed regions
     */
    private BufferedImage extractChanges(BufferedImage img, boolean[][] mask) {
        int width = img.getWidth();
        int height = img.getHeight();
        BufferedImage result = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x]) {
                    result.setRGB(x, y, img.getRGB(x, y));
                } else {
                    result.setRGB(x, y, 0); // Black
                }
            }
        }
        return result;
    }

    /**
     * Helper methods
     */
    private BufferedImage resizeImage(BufferedImage img, int width, int height) {
        BufferedImage resized = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.drawImage(img, 0, 0, width, height, null);
        g.dispose();
        return resized;
    }

    private BufferedImage copyImage(BufferedImage img) {
        BufferedImage copy = new BufferedImage(img.getWidth(), img.getHeight(), img.getType());
        Graphics2D g = copy.createGraphics();
        g.drawImage(img, 0, 0, null);
        g.dispose();
        return copy;
    }

    private int countTruePixels(boolean[][] mask) {
        int count = 0;
        for (boolean[] row : mask) {
            for (boolean pixel : row) {
                if (pixel) count++;
            }
        }
        return count;
    }

    /**
     * Inner classes
     */
    static class Contour {
        List<Point> points = new ArrayList<>();
        int area = 0;
        int minX = Integer.MAX_VALUE;
        int maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE;
        int maxY = Integer.MIN_VALUE;
    }

    static class ChangeResult {
        public BufferedImage visualization;
        public BufferedImage changesOnly;
        public boolean[][] mask;
        public int totalChanges;
        public int changedPixels;
        public int totalPixels;
        public double changePercentage;

        public ChangeResult(BufferedImage visualization, BufferedImage changesOnly,
                            boolean[][] mask, int totalChanges, int changedPixels,
                            int totalPixels, double changePercentage) {
            this.visualization = visualization;
            this.changesOnly = changesOnly;
            this.mask = mask;
            this.totalChanges = totalChanges;
            this.changedPixels = changedPixels;
            this.totalPixels = totalPixels;
            this.changePercentage = changePercentage;
        }
    }
}
