import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

import java.nio.file.Paths;
import java.nio.file.Path;

boolean doDeepVision = true;
DeepVision vision;
Pix2PixNetwork network;
int dim = 256;

void modelSetup() {
  vision = new DeepVision(this);
  
  String url = sketchPath(new File("data", "pix2pix002_140_net_G.onnx").getPath());
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
