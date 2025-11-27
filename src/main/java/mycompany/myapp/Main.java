package mycompany.myapp;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;


public class Main {
    private final JFrame frame = new JFrame("Webcam Viewer");
    private final JLabel imageLabel = new JLabel();
    private final JButton toggleButton = new JButton("Start");
    private final JButton captureButton = new JButton("Capture");
    private final JButton captureButton2 = new JButton("Capture 2");
    private final JButton compareButton = new JButton("Compare");
    private final AtomicBoolean running = new AtomicBoolean(false);

    private OpenCVFrameGrabber grabber;
    private SwingWorker<Void, BufferedImage> worker;
    private final Java2DFrameConverter converter = new Java2DFrameConverter();

    //for comparison
    private BufferedImage image1;
    private BufferedImage image2;

    public Main(){
        imageLabel.setHorizontalAlignment(SwingConstants.CENTER);
        imageLabel.setPreferredSize(new Dimension(640, 480));

        toggleButton.addActionListener(e -> {
            if(running.get()) stopCapture();
            else startCapture();
        });

        captureButton.addActionListener(e -> {
           new Thread(()->{
               if(isGrabberRunning()){
                   try{
                       Frame frame = grabber.grab();
                       if(frame != null){
                           BufferedImage image = converter.convert(frame);
                           if(image != null){
                               ImageIO.write(image, "jpg", new File("Capture.jpeg"));
                               image1 = image;
                           }
                       }
                   }
                   catch(Exception ex){
                       System.out.println(ex.getMessage());
                   }
               }
               else System.out.println("Grabber isn't running");

           }).start();
        });

        captureButton2.addActionListener(e -> {
           new Thread(()->{
                if(isGrabberRunning()){
                    try{
                        Frame frame = grabber.grab();
                        if(frame != null){
                            BufferedImage image = converter.convert(frame);
                            if(image != null){
                                ImageIO.write(image, "jpg", new File("Capture2.jpeg"));
                                image2 = image;
                            }
                        }
                    }
                    catch(Exception ex){
                        System.out.println(ex.getMessage());
                    }
                }
                else{
                    System.out.println("Grabber isn't running");
                }
           }).start();
        });

        compareButton.addActionListener(e -> {
            new Thread(()->{
                if(image1 != null && image2 != null){
                    try{
                        ChangeDetector comparator = new ChangeDetector(25, 100,true);
                        ChangeDetector.ChangeResult result = comparator.detectChanges("Capture.jpeg", "Capture2.jpeg");

                        // Save results
                        ImageIO.write(result.visualization, "jpg", new File("changes_detected.jpg"));
                        ImageIO.write(result.changesOnly, "jpg", new File("changes_only.jpg"));

                        // Print statistics
                        System.out.println("Changes detected: " + result.totalChanges);
                        System.out.printf("Changed area: %.2f%%\n", result.changePercentage);
                        System.out.println("Changed pixels: " + result.changedPixels);
                    }
                    catch(Exception ex){
                        System.out.println(ex.getMessage());
                    }
                }
            }).start();
        });

        JPanel control = new  JPanel();
        control.add(toggleButton);
        control.add(captureButton);
        control.add(captureButton2);
        control.add(compareButton);

        frame.setLayout(new BorderLayout());
        frame.add(imageLabel, BorderLayout.CENTER);
        frame.add(control, BorderLayout.SOUTH);
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);

        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent e) {
                File capture1 = new File("Capture.jpeg");
                capture1.delete();
                File capture2 = new File("Capture2.jpeg");
                capture2.delete();
                stopCapture();
            }
        });
    }

    private void startCapture(){
        running.set(true);
        toggleButton.setText("Stop");
        worker = new SwingWorker<>() {
            @Override
            protected Void doInBackground(){
                try{
                    grabber = new OpenCVFrameGrabber(0);
                    grabber.start();
                    while(running.get() && !isCancelled()){
                        Frame frame = grabber.grab();
                        if(frame == null) break;
                        BufferedImage image = converter.convert(frame);
                        if (image != null) publish(image);
                        Thread.sleep(30);
                    }
                }
                catch(FrameGrabber.Exception | InterruptedException ex){
                    System.out.println(ex.getMessage());
                }
                finally{
                    try{
                        if(grabber != null) grabber.stop();
                    }
                    catch(FrameGrabber.Exception ignored) {}
                }
                return null;
            }

            @Override
            protected void process(java.util.List<BufferedImage> chunks){
                //show the latest frame
                BufferedImage latest = chunks.getLast();
                imageLabel.setIcon(new ImageIcon(latest));
            }

            @Override
            protected void done(){
                running.set(false);
                toggleButton.setText("Start");
            }
        };
        worker.execute();
    }

    private void stopCapture(){
        running.set(false);
        toggleButton.setText("Stopping...");
        if(worker != null) worker.cancel(true);
    }

    public void show(){
        SwingUtilities.invokeLater(() -> frame.setVisible(true));
    }

    public boolean isGrabberRunning() {
        if (running.get()) return true;
        // fallback: check that a grabber exists and a worker is still executing
        return grabber != null && worker != null && !worker.isDone();
    }

    static void main() {
        new Main().show();
    }
}
