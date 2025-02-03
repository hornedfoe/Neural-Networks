package JAVA;
public class Loss {
    private String type;

    public Loss(String type) {
        this.type = type;
    }

    public double calculateLoss(double actual, double predicted) {
        switch (type) {
            case "Regression":
                return mean_square_error(actual, predicted);
            case "Two_class_Classification" , "Classification":
                return cross_entropy(actual, predicted);

            default:
                throw new IllegalAccessError(type + " is not a valid type");
        }
    }

    public double[] derivative(double[] actual, double[] predicted) {
        switch (type) {
            case "Regression":
                return derivative_mean_square_error(actual, predicted);
            case "Two_class_Classification" , "Classification":
                return derivative_cross_entropy(actual, predicted);
            default:
                throw new IllegalAccessError(type + " is not a valid type");
        }
    }

    private double mean_square_error(double actual, double predicted) {
        return Math.pow((actual - predicted), 2);
    }

    private double cross_entropy(double actual, double predicted) {
        return -(actual * Math.log(predicted) + (1 - actual) * Math.log(1 - predicted));
    }

    private double[] derivative_cross_entropy(double[] actual, double[] predicted) {
        double[] ret = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            predicted[i] = Math.max(predicted[i], 1e-15);
            ret[i] = ((1 - actual[i]) / (1 - predicted[i])) - (actual[i] / predicted[i]);
        }
        return ret;
    }

    private double[] derivative_mean_square_error(double[] actual, double[] predicted) {
        double[] ret = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            ret[i] = 2 * (predicted[i] - actual[i]);
        }
        return ret;
    }

    public double calculateR2(double[][] Y, double[][] predictions) {
        double mean = calculateMean(Y);
        double tss = calculateTSS(Y, mean);
        double rss = calculateRSS(Y, predictions);
        double r2 = 1 - (rss / tss);
        return r2;
    }

    private double calculateMean(double[][] array) {
        double sum = 0.0;
        for (int i = 0; i < array.length; i++) {
            sum += array[i][0];
        }
        return sum / array.length;
    }

    private double calculateTSS(double[][] array, double mean) {
        double tss = 0.0;
        for (int i = 0; i < array.length; i++) {
            tss += Math.pow(array[i][0] - mean, 2);
        }
        return tss;
    }

    private double calculateRSS(double[][] actual, double[][] predictions) {
        double rss = 0.0;
        for (int i = 0; i < actual.length; i++) {
            rss += Math.pow(actual[i][0] - predictions[i][0], 2);
        }
        return rss;
    }
}
