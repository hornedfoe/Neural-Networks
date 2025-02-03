package JAVA;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.io.File;
import java.io.IOException;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

public class Main {

    private static double[] function(double[] X) {
        double x = X[0];
        if (x > 50) {
            x = x - 50;
        }
        x = Math.sin(x);
        return new double[]{x};
    }

    public static void main(String[] args) throws IllegalAccessException, IOException {
        if (args.length != 1) {
            System.out.println("Usage: java JAVA.Main <path to parameters.xml>");
            return;
        }

        String xmlFilePath = args[0];
        Parameters params = readParametersFromXML(xmlFilePath);

        Other.deleteAll(new File("PYTHON/images"));

        boolean video = true && params.feature == 1 && params.output == 1;

        Random rand = new Random();

        Model model = new Model(params.type, params.structure, params.functions, params.epochs, params.lr, params.Model,
                params.beta1, params.beta2, params.epsilon, params.batchSize, params.lambda, video);

        double[][] X = new double[params.train][params.feature];
        double[][] Y = new double[params.train][params.output];

        double[][] x = new double[params.test][params.feature];
        double[][] y = new double[params.test][params.output];

        for (int i = 0; i < params.train; i++) {
            for (int j = 0; j < params.feature; j++) {
                X[i][j] = rand.nextDouble(params.lower_bound, params.upper_bound);
            }
            Y[i] = function(X[i]);
        }

        for (int i = 0; i < params.test; i++) {
            for (int j = 0; j < params.feature; j++) {
                x[i][j] = rand.nextDouble(params.lower_bound, params.upper_bound);
            }
            y[i] = function(x[i]);
        }

        Other.write(X, Y, "PYTHON/io.txt");

        double[][] vX = new double[params.validate][params.feature];
        double[][] vY = new double[params.validate][params.output];
        double[][] predictedY = new double[params.validate][params.output];

        for (int i = 0; i < params.validate; i++) {
            for (int j = 0; j < params.feature; j++) {
                vX[i][j] = rand.nextDouble(params.lower_bound, params.upper_bound);
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
                Other.runPy(new String[]{"python", "PYTHON/plot.py", "PYTHON/io.txt", "Actual"});
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
            Other.runPy(new String[]{"python", "PYTHON/plot.py", "PYTHON/predicted.txt", "Predicted"});

        if (video)
            Other.runPy(new String[]{"python", "PYTHON/compile.py"});
    }

    private static Parameters readParametersFromXML(String xmlFilePath) {
        Parameters params = new Parameters();
        try {
            File xmlFile = new File(xmlFilePath);
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(xmlFile);

            doc.getDocumentElement().normalize();

            Element root = doc.getDocumentElement();

            params.feature = Integer.parseInt(root.getElementsByTagName("feature").item(0).getTextContent());
            params.output = Integer.parseInt(root.getElementsByTagName("output").item(0).getTextContent());
            params.train = Integer.parseInt(root.getElementsByTagName("train").item(0).getTextContent());
            params.test = Integer.parseInt(root.getElementsByTagName("test").item(0).getTextContent());
            params.validate = Integer.parseInt(root.getElementsByTagName("validate").item(0).getTextContent());
            params.lower_bound = Double.parseDouble(root.getElementsByTagName("lower_bound").item(0).getTextContent());
            params.upper_bound = Double.parseDouble(root.getElementsByTagName("upper_bound").item(0).getTextContent());

            params.structure = Arrays.stream(root.getElementsByTagName("structure").item(0).getTextContent().split(","))
                    .map(Integer::parseInt).toArray(Integer[]::new);

            params.functions = root.getElementsByTagName("functions").item(0).getTextContent().split(",");

            params.type = root.getElementsByTagName("type").item(0).getTextContent();
            params.epochs = Integer.parseInt(root.getElementsByTagName("epochs").item(0).getTextContent());
            params.lr = Double.parseDouble(root.getElementsByTagName("lr").item(0).getTextContent());
            params.Model = root.getElementsByTagName("Model").item(0).getTextContent();
            params.beta1 = Double.parseDouble(root.getElementsByTagName("beta1").item(0).getTextContent());
            params.beta2 = Double.parseDouble(root.getElementsByTagName("beta2").item(0).getTextContent());
            params.epsilon = Double.parseDouble(root.getElementsByTagName("epsilon").item(0).getTextContent());
            params.batchSize = Integer.parseInt(root.getElementsByTagName("batchSize").item(0).getTextContent());
            params.lambda = Double.parseDouble(root.getElementsByTagName("lambda").item(0).getTextContent());
        } catch (Exception e) {
            e.printStackTrace();
        }
        return params;
    }

    private static class Parameters {
        int feature;
        int output;
        int train;
        int test;
        int validate;
        double lower_bound;
        double upper_bound;
        Integer[] structure;
        String[] functions;
        String type;
        Integer epochs;
        double lr;
        String Model;
        double beta1;
        double beta2;
        double epsilon;
        Integer batchSize;
        double lambda;
    }
}
