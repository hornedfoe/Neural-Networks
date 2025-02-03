package JAVA;
import java.lang.Math;
import java.util.Arrays;

public class Activation {
    Integer[] structure;
    String[] functions;

    public Activation(Integer[] structure, String[] functions) {
        this.structure = structure;
        this.functions = functions;
    }

    public double[] activate(double[] x, Integer i) {
        String activator = functions[i];
        switch (activator) {
            case "linear", "input":
                return identity(x);
            case "reLU":
                return reLU(x);
            case "tanh":
                return tanh(x);
            case "sigmoid":
                return sigmoid(x);
            default:
                throw new IllegalArgumentException("Unsupported activation function: " + activator);
        }
    }

    public double[] differentiate(double[] x, Integer i) {
        String activator = functions[i];
        switch (activator) {
            case "linear", "input":
                return d_identity(x);
            case "reLU":
                return d_reLU(x);
            case "tanh":
                return d_tanh(x);
            case "sigmoid":
                return d_sigmoid(x);
            default:
                throw new IllegalAccessError(activator + " not a valid function");
        }
    }

    private double[] identity(double[] x) {
        double[] ret = new double[x.length];
        for(int i = 0 ; i < x.length ; i++) ret[i] = x[i];
        return ret;
    }

    private double[] d_identity(double[] x) {
        double[] ret = new double[x.length];
        Arrays.fill(ret, 1d);
        return ret;
    }

    private double[] reLU(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = Math.max(0d, x[i]);
        }
        return ret;
    }

    private double[] d_reLU(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = x[i] > 0 ? 1d : 0d;
        }
        return ret;
    }

    private double[] sigmoid(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = 1d / (1 + Math.exp(-x[i]));
        }
        return ret;
    }

    private double[] d_sigmoid(double[] x) {
        double ret[] = sigmoid(x);
        for (int i = 0; i < x.length; i++) {
            ret[i] = ret[i] * (1 - ret[i]);
        }
        return ret;
    }

    private double[] tanh(double[] x) {
        double[] ret = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            ret[i] = Math.tanh(x[i]);
        }
        return ret;
    }

    private double[] d_tanh(double[] x) {
        double ret[] = tanh(x);
        for (int i = 0; i < x.length; i++) {
            ret[i] = 1 / Math.pow(Math.cosh(ret[i]) , 2);
        }
        return ret;
    }
}
