package ch.bildspur.vision;

import ch.bildspur.vision.network.BaseNeuralNetwork;
import ch.bildspur.vision.result.ImageResult;
import ch.bildspur.vision.util.CvProcessingUtils;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import processing.core.PImage;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class Pix2PixNetwork extends BaseNeuralNetwork<ImageResult> {
  private Path model;
  private Net net;
  private int dim = 256;
  
  public Pix2PixNetwork(Path model) {
    this.model = model;
  }

  @Override
  public boolean setup() {
    net = readNetFromONNX(model.toAbsolutePath().toString());

    DeepVision.enableDesiredBackend(net);

    return true;
  }

  @Override
  public ImageResult run(Mat frame) {
    System.out.println("Input image size: " + frame.size().width() + ", " + frame.size().height());

    // convert image into batch of images
    Mat inputBlob = blobFromImage(frame, 1.0 / 255.0, new Size(256, 256), new Scalar(0, 0, 0, 0), false, false, CV_32F);

    // set input
    net.setInput(inputBlob);

    // create output layers
    StringVector outNames = net.getUnconnectedOutLayersNames();
    MatVector outs = new MatVector(outNames.size());

    // run detection
    net.forward(outs, outNames);
    Mat output = outs.get(0);
    
    // reshape output mat
    // TODO why does this tile the image
    output = output.reshape(0, 768);
    System.out.println("Output orig size: " + output.size(4) + ", " +  + output.size(3) + ", " + output.size(2) + ", " + output.size(1) + ", " + output.size(0));

    // resize output instead of PImage to avoid Processing4 problems
    resize(output, output, new Size(dim, dim));
    System.out.println("Output resized: " + output.size(1) + ", " +  + output.size(0));

    // todo: result a depth frame instead of a color image!
    PImage result = new PImage(dim, dim);
    matToImage(output, result);
    result = result.get(0, 0, result.width, result.height/3);
    result.resize(dim, dim);
    return new ImageResult(result);
  }

  private void matToImage(Mat mat, PImage img) {
      // find min / max
      DoublePointer minValuePtr = new DoublePointer(1);
      DoublePointer maxValuePtr = new DoublePointer(1);

      minMaxLoc(mat, minValuePtr, maxValuePtr, null, null, null);

      double minValue = minValuePtr.get();
      double maxValue = maxValuePtr.get();

      double distance = maxValue - minValue;
      double minScaled = minValue / distance;

      double alpha = 1.0 / distance * 255.0;
      double beta = -1.0 * minScaled * 255.0;
      System.out.println("alpha: " + alpha + ", beta: " + beta);
      
      mat.convertTo(mat, CV_8U, alpha, beta);
      CvProcessingUtils.toPImage(mat, img);
  }

  public Net getNet() {
    return net;
  }
}
