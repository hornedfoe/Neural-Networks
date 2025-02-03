package JAVA;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.io.File;
import java.io.IOException;

public class Main {

    private static double[] function(double[] X) {
        double x = X[0];
        if(x > 50){
            x = x - 50;
        }
        x = Math.sin(x);
        return new double[] { x };
    }

    public static void main(String[] args) throws IllegalAccessException, IOException {

        Other.deleteAll(new File("PYTHON/images"));

        // Initializer.setSeed(42);

        int feature = 1;
        int output = 1;

        boolean video = !true && feature == 1 && output == 1;

        int train = 3000;
        int test = train / 5;
        int validate = train / 10;

        double lower_bound = -2, upper_bound = 20;

        Random rand = new Random();

        Integer[] structure = { feature, 30, 30, 30, 30, output };
        String[] functions = { "linear", "reLU", "reLU", "reLU","reLU","linear" };

        String type = "Regression";
        Integer epochs = 1000;
        double lr = 0.01d;
        String Model = "adam";
        double beta1 = 0.99d;
        double beta2 = 0.999d;
        double epsilon = 1e-8d;
        Integer batchSize = train;
        double lambda = 1e-18d;

        Model model = new Model(type, structure, functions, epochs, lr, Model, beta1,
                beta2, epsilon, batchSize,
                lambda, video);

        // Model model = new Model("C:\\Users\\horne\\Desktop\\Neural Networks\\wb.pt");

        double[][] X = new double[train][feature];
        double[][] Y = new double[train][output];

        double[][] x = new double[test][feature];
        double[][] y = new double[test][output];

        for (int i = 0; i < train; i++) {
            for (int j = 0; j < feature; j++) {
                X[i][j] = rand.nextDouble(lower_bound, upper_bound);
            }
            Y[i] = function(X[i]);
        }

        for (int i = 0; i < test; i++) {
            for (int j = 0; j < feature; j++) {
                x[i][j] = rand.nextDouble(lower_bound, upper_bound);
            }
            y[i] = function(x[i]);
        }

        Other.write(X, Y, "PYTHON/io.txt");

        double[][] vX = new double[validate][feature];
        double[][] vY = new double[validate][output];
        double[][] predictedY = new double[validate][output];

        for (int i = 0; i < validate; i++) {
            for (int j = 0; j < feature; j++) {
                vX[i][j] = rand.nextDouble(lower_bound, upper_bound);
            }
            vY[i] = function(vX[i]);
        }

        ExecutorService executorService = Executors.newFixedThreadPool(2);

        Runnable trainTask = () -> {
            try {
                process(model, X, Y, x, y, vX, vY, predictedY, video);
            } catch (Exception e) {
                System.err.println("Error in training task: " + e);
                e.printStackTrace();
            }
        };

        Runnable plotTask = () -> {
            try {
                Other.runPy(new String[] { "python", "PYTHON/plot.py", "PYTHON/io.txt", "Actual" });
            } catch (Exception e) {
                System.err.println("Error in plot task: " + e);
                e.printStackTrace();
            }
        };

        executorService.submit(trainTask);
        if (X[0].length == 1 && Y[0].length == 1)
            executorService.submit(plotTask);

        executorService.shutdown();

    }

    private static void process(Model model, double[][] X, double[][] Y, double[][] x, double[][] y, double[][] vX,
            double[][] vY, double[][] predictedY, boolean video) {

        model.train(X, Y, x, y);

        predictedY = model.predict(vX);

        double R2 = model.calculateR2(vY, predictedY);

        for (int i = 0; i < vX.length; i++) {
            System.out.println(
                    "\u001B[34m Features:\u001B[33m" + Arrays.toString(vX[i]) + "\u001B[34m Predicted = \u001B[33m"
                            + Arrays.toString(predictedY[i]) + "\u001B[34m Actual= \u001B[33m" + Arrays.toString(vY[i])
                            + "\u001B[0m");
        }

        System.out.println(
                "\u001B[1;35m ========================================================================================="
                        + "\u001B[0m");

        System.out.println("\u001B[1;31m Validate Evaluation : " + "\u001B[0m");
        System.out.println("\u001B[34m R2: \u001B[1;33m" + R2 + "\u001B[0m");

        System.out.println(
                "\u001B[1;35m ========================================================================================="
                        + "\u001B[0m");

        model.export();

        if (X[0].length == 1 && Y[0].length == 1)
            Other.write(X, model.predict(X), "PYTHON/predicted.txt");

        if (X[0].length == 1 && Y[0].length == 1)
            Other.runPy(new String[] { "python", "PYTHON/plot.py", "PYTHON/predicted.txt", "Predicted" });

        if (video)
            Other.runPy(new String[] { "python", "PYTHON/compile.py" });

    }
}




