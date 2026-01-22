import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;
import ch.bildspur.vision.dependency.*;

import java.nio.file.Paths;
import java.nio.file.Path;

DeepVision vision;
Pix2PixNetwork network;

String[] urls = {
  "contour_pix2pix_195_net_G.onnx",
  "contour_reverse_pix2pix_195_net_G.onnx",
  "latest_net_G.onnx",
  "latest_net_G_simplified.onnx",
  "contour_pix2pix_195_net_G.onnx",
  "new_pix2pix002_140_net_G_simplified.onnx"
};

String url = urls[1];

void modelSetup() {
  vision = new DeepVision(this);
  
  url = sketchPath(new File("data", url).getPath());
  
  println("Loading model from " + url);
  Path model = Paths.get(url).toAbsolutePath();
  network = new Pix2PixNetwork(model);
  network.setup();
}

PImage modelInference(PImage img) { 
  println("Inferencing...");
  ImageResult result = network.run(img);
  PImage returnImg = result.getImage();
  println("...done!"); 
  
  return returnImg;
}
