PImage img;
PImage result;

void setup() {
  size(50, 50);
  img = loadImage("test1.jpg");
  //img = loadImage("test2.jpg");
  //img = loadImage("test3.jpg");
  surface.setSize(img.width*2, img.height*2);
  
  modelSetup();
  result = modelInference(img);
}

void draw() {
  if (drawResult) {
    image(result, 0, 0, width, height);
  } else {
    image(img, 0, 0, width, height);
  }
}
