package JAVA;

public class ExternalObserver {
    private Model model;
    private boolean video;
    private int capture_count;
    private int epoch;
    
    public void update(Model model, boolean video, int capture_count){
        this.model = model;
        this.video = video;
        this.capture_count = capture_count;
        this.epoch = 0;
    }
    public boolean ping(int epoch){
        this.epoch = epoch;
        if(!video || epoch % capture_count != 0) return false;
        return true;
    }
    public void run(double[][] X){
        Other.write(X, model.predict(X), "PYTHON/images/images.txt");
        Other.runPy(new String[] { "python", "PYTHON/generate.py", "PYTHON/images/images.txt",
                        "PYTHON/images/image" + epoch + ".png", "epoch " + epoch });
    }
}