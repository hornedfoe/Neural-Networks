package JAVA;
public class Optimizer {

    private String optimizer;
    private double lr = 0.001d;
    private double beta1 = 0.99d, beta2 = 0.999d;
    private double[][] SWD[], SBD;
    private double[][] VWD[], VBD;
    private double epsilon = 1e-8d;

    public Optimizer(String optimizer, Integer[] structure, double lr, double beta1, double beta2, double epsilon) {
        this.optimizer = optimizer;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;

        this.SWD = new double[structure.length - 1][][];
        this.VWD = new double[structure.length - 1][][];

        this.SBD = new double[structure.length][];
        this.VBD = new double[structure.length][];

        for (int i = 0; i < structure.length - 1; i++) {
            this.SWD[i] = new double[structure[i]][structure[i + 1]];
            this.VWD[i] = new double[structure[i]][structure[i + 1]];
            for (int j = 0; j < structure[i]; j++) {
                for (int k = 0; k < structure[i + 1]; k++) {
                    this.SWD[i][j][k] = 0d;
                    this.VWD[i][j][k] = 0d;
                }
            }
        }

        for (int i = 0; i < structure.length; i++) {
            this.SBD[i] = new double[structure[i]];
            this.VBD[i] = new double[structure[i]];
            for (int j = 0; j < structure[i]; j++) {
                this.SBD[i][j] = 0d;
                this.VBD[i][j] = 0d;
            }
        }

    }

    public double compute(double W, double dW, Integer i, Integer j, Integer k) {
        switch (optimizer) {
            case "gradient_descent":
                return gradient_descent(W, dW);
            case "momentum":
                return momentum(W, dW, i, j, k);
            case "rms_prop":
                return rms_prop(W, dW, i, j, k);
            case "adam":
                return adam(W, dW, i, j, k);
            default:
                throw new IllegalArgumentException("Unsupported optimizer: " + optimizer);
        }
    }

    public double compute(double B, double dB, Integer i, Integer j) {
        switch (optimizer) {
            case "gradient_descent":
                return gradient_descent(B, dB);
            case "momentum":
                return momentum(B, dB, i, j);
            case "rms_prop":
                return rms_prop(B, dB, i, j);
            case "adam":
                return adam(B, dB, i, j);
            default:
                throw new IllegalArgumentException("Unsupported optimizer: " + optimizer);
        }
    }

    private double gradient_descent(double x, double dx) {
        return x - lr * dx;
    }

    private double momentum(double W, double dW, Integer i, Integer j, Integer k) {
        VWD[i][j][k] = VWD[i][j][k] * beta1 + dW * (1 - beta1);
        return W - lr * VWD[i][j][k];
    }

    private double momentum(double B, double dB, Integer i, Integer j) {
        VBD[i][j] = VBD[i][j] * beta1 + dB * (1 - beta1);
        return B - lr * VBD[i][j];
    }

    private double rms_prop(double W, double dW, Integer i, Integer j, Integer k) {
        SWD[i][j][k] = SWD[i][j][k] * beta2 + dW * dW * (1 - beta2);
        return W - lr * dW / Math.sqrt(SWD[i][j][k] + epsilon);
    }

    private double rms_prop(double B, double dB, Integer i, Integer j) {
        SBD[i][j] = SBD[i][j] * beta2 + dB * dB * (1 - beta2);
        return B - lr * dB / Math.sqrt(SBD[i][j] + epsilon);
    }

    private double adam(double W, double dW, Integer i, Integer j, Integer k) {
        SWD[i][j][k] = SWD[i][j][k] * beta2 + dW * dW * (1 - beta2);
        VWD[i][j][k] = VWD[i][j][k] * beta1 + dW * (1 - beta1);
        return W - lr * VWD[i][j][k] / Math.sqrt(SWD[i][j][k] + epsilon);
    }

    private double adam(double B, double dB, Integer i, Integer j) {
        SBD[i][j] = SBD[i][j] * beta2 + dB * dB * (1 - beta2);
        VBD[i][j] = VBD[i][j] * beta1 + dB * (1 - beta1);
        return B - lr * VBD[i][j] / Math.sqrt(SBD[i][j] + epsilon);
    }
}
