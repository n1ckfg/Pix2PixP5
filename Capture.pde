Capture cam;

void captureSetup() {
  captureInit(640, 480, 0, 30);
}

void captureSetup(int device) {
  captureInit(640, 480, device, 30);
}

void captureSetup(int w, int h, int device, int fps) {
  captureInit(w, h, device, fps);
}

void captureInit(int w, int h, int device, int fps) {
  String[] cameras = Capture.list();
  printArray(cameras);
  
  if (isMacOS()) {
    cam = new Capture(this, w, h, "pipeline:avfvideosrc device-index=" + device, fps);    
  } else {
    cam = new Capture(this, w, h, cameras[device], fps);
  }
  
  cam.start();  
}

boolean isMacOS() {
  String os = System.getProperty("os.name").toLowerCase();
  return os.contains("mac");
}
