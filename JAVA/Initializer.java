package JAVA;

import java.util.Random;

public class Initializer {

    private static Random rand = new Random();

    public static void setSeed(int seed){
        rand = new Random(seed);
    }

    
    public static double[][] initialize(Integer m, Integer n, String function) throws IllegalAccessException {
        switch (function) {
            case "sigmoid", "tanh", "linear": {
                return xavier(m, n);
            }
            case "reLU": {
                return xavier(m, n);
            }
            default:
                throw new IllegalAccessException(function + " is not a valid activation function");
        }
    }

    public static double[] initialize(Integer n, String function) throws IllegalAccessException {
        switch (function) {
            case "sigmoid", "tanh", "linear": {
                return xavier(n);
            }
            case "reLU": {
                return xavier(n);
            }
            default:
                throw new IllegalAccessException(function + " is not a valid activation function");
        }
    }

    private static double[][] xavier(Integer m, Integer n) {
        double ret[][] = new double[m][n];
        double limit = Math.sqrt(6d / (m + n));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ret[i][j] = rand.nextDouble(0 , 1) * limit * 2 - limit;
            }
        }
        return ret;
    }

    private static double[] xavier(Integer n) {
        double[] ret = new double[n];
        double limit = Math.sqrt(2d / n);
        for (int i = 0; i < n; i++) {
            ret[i] = rand.nextDouble(0 , 1) * limit * 2 - limit;
        }
        return ret;
    }

    private static double[][] he(Integer m, Integer n) {
        double ret[][] = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ret[i][j] = Math.sqrt(2.0 / (m)) * rand.nextDouble(0 , 1);
            }
        }
        return ret;
    }

    private static double[] he(Integer n) {
        double[] ret = new double[n];
        for (int i = 0; i < n; i++) {
            ret[i] = Math.sqrt(2.0 / n) * rand.nextDouble(0 , 1);
        }
        return ret;
    }
}