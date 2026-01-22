import processing.javafx.*;
import processing.video.Capture;

PImage img;
PImage result;
Capture cam;
boolean useVideo = true;

void setup() {
  size(640, 480, FX2D);
  //colorMode(HSB, 360, 100, 100);

  String[] cameras = Capture.list();
 
  cam = new Capture(this, 640, 480, cameras[1], 30);
  cam.start();
  
  /*
  if (useVideo) {
    surface.setSize(512, 512);
  } else {
    img = loadImage("test1.jpg");
    //img = loadImage("test2.jpg");
    //img = loadImage("test3.jpg");
    surface.setSize(img.width*2, img.height*2);
  }
  */
  
  modelSetup();
  
  if (!useVideo) result = modelInference(img);
}

void draw() {
  if (useVideo) {
    if (cam.available()) {
      cam.read();
      result = modelInference(cam);
    } else if (cam.width == 0) {
      return;
    }
    image(result, 0, 0, width, height);
  } else {
    if (drawResult) {
      image(result, 0, 0, width, height);
    } else {
      image(img, 0, 0, width, height);
    }
  }

  surface.setTitle("FPS: " + Math.round(frameRate));
}
