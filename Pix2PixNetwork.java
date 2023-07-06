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
    Mat inputBlob = blobFromImage(frame, 1.0 / 255.0, new Size(dim, dim), new Scalar(0, 0, 0, 0), false, false, CV_32F);

    // set input
    net.setInput(inputBlob);

    // create output layers
    StringVector outNames = net.getUnconnectedOutLayersNames();
    MatVector outs = new MatVector(outNames.size());

    // run detection
    net.forward(outs, outNames);
    Mat output = outs.get(0);
    
    // reshape output mat
    System.out.println("Output size raw: " + output.size().width() + ", " + output.size().height());
    output = output.reshape(0, 768);
    System.out.println("Output size reshaped: " + output.size().width() + ", " + output.size().height());
           
    PImage result = new PImage(dim, dim*3);
    matToImage(output, result);
    
    PImage redChannel = result.get(0, 0, 256, 256);
    redChannel.loadPixels();
    PImage greenChannel = result.get(0, 256, 256, 256);
    greenChannel.loadPixels();
    PImage blueChannel = result.get(0, 512, 256, 256);
    blueChannel.loadPixels();

    PImage returns = new PImage(dim, dim);
    returns.loadPixels();    
    for (int i=0; i<returns.pixels.length; i++) {
      int r = red(redChannel.pixels[i]);
      int g = green(greenChannel.pixels[i]);
      int b = blue(blueChannel.pixels[i]);
      int a = alpha(returns.pixels[i]);
      returns.pixels[i] = color(r, g, b, a);
    }
    returns.updatePixels();
   
    return new ImageResult(returns);
  }
 
  private void matToImage(Mat mat, PImage img) {
    mat = multiply(mat, 0.5).asMat();
    mat = add(mat, new Scalar(0.5)).asMat();
    mat = multiply(mat, 255).asMat();
    
    mat.convertTo(mat, CV_8U);
    CvProcessingUtils.toPImage(mat, img);
  }

  public Net getNet() {
    return net;
  }
  
  private int color(int r, int g, int b, int a) {
    return (a << 24) | (r << 16) | (g << 8) | b;
  }

  private int red(int c) {
    return (c >> 16) & 0xFF;
  }
  
  private int green(int c) {
    return (c >> 8) & 0xFF;
  }
  
  private int blue(int c) {
    return c & 0xFF;
  }

  private int alpha(int c) {
    return (c >> 24) & 0xFF;
  }
  
}
