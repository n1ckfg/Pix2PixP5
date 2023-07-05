import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

import java.nio.file.Paths;
import java.nio.file.Path;

boolean doDeepVision = true;
DeepVision vision;
Pix2PixNetwork network;
int dim = 256;
//String url = "pix2pix001_140_net_G.onnx";
//String url = "pix2pix002_140_net_G.onnx";
//String url = "pix2pix003_140_net_G.onnx";
//String url = "pix2pix004_140_net_G.onnx";
String url = "contour_pix2pix_195_net_G.onnx";
//String url = "contour_reverse_pix2pix_140_net_G.onnx";

void modelSetup() {
  vision = new DeepVision(this);
  
  url = sketchPath(new File("data", url).getPath());
  println("Loading model from " + url);
  Path model = Paths.get(url).toAbsolutePath();

  network = new Pix2PixNetwork(model);
  println("Loading model...");
  network.setup();
}

PImage modelInference(PImage img) { 
  println("Inferencing...");
  ImageResult result = network.run(img);
  println("...done!");
  
  PImage returnImg = result.getImage();
  returnImg.resize(dim, dim);
  
  return returnImg;
}
