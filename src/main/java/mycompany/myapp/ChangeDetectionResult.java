package mycompany.myapp;

import org.bytedeco.opencv.opencv_core.Mat;

public class ChangeDetectionResult {
    private final Mat visualization;
    private final Mat mask;
    private final Mat changesOnly;
    private final Mat diffImage;
    private final Mat thresholdImage;
    private final ChangeStats stats;

    public ChangeDetectionResult(Mat visualization, Mat mask, Mat changesOnly, Mat diffImage, Mat thresholdImage, ChangeStats stats){
        this.visualization = visualization;
        this.mask = mask;
        this.changesOnly = changesOnly;
        this.diffImage = diffImage;
        this.thresholdImage = thresholdImage;
        this.stats = stats;
    }

    public Mat getVisualization() { return visualization; }
    public Mat getMask() { return mask; }
    public Mat getChangesOnly() { return changesOnly; }
    public Mat getDiffImage() { return diffImage; }
    public Mat getThresholdImage() { return thresholdImage; }
    public ChangeStats getStats() { return stats; }

}
