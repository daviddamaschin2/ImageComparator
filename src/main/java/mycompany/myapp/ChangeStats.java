package mycompany.myapp;

public class ChangeStats {
    private final int totalChanges;
    private final long changedPixels;
    private final long totalPixels;
    private final double changedPercentage;

    public ChangeStats(int totalChanges, long changedPixels, long totalPixels, double changedPercentage) {
        this.totalChanges = totalChanges;
        this.changedPixels = changedPixels;
        this.totalPixels = totalPixels;
        this.changedPercentage = changedPercentage;
    }

    public int getTotalChanges() { return totalChanges; }
    public long getChangedPixels() { return changedPixels; }
    public long getTotalPixels() { return totalPixels; }
    public double getChangePercentage() { return changedPercentage; }
}
