PImage img;
PImage result;

void setup() {
  size(50, 50);
  img = loadImage("test1.jpg");
  //img = loadImage("test2.jpg");
  //img = loadImage("test3.jpg");
  surface.setSize(img.width, img.height);
  
  modelSetup();
  result = modelInference(img);
}

void draw() {
  if (drawResult) {
    image(result, 0, 0);
  } else {
    image(img, 0, 0);
  }
}
