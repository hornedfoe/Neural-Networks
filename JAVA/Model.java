package JAVA;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class Model {    
    private double[][][] W;
    private double[][] B;
    private String[] functions;
    private Integer epochs;
    private Integer[] structure;
    private Optimizer optimizer;
    private String opt;
    private Integer batchSize = 32;
    private Activation activation;
    private double lambda;
    private Loss cost;
    private String type;
    private Double lr;
    private double beta1;
    private double beta2;
    private double epsilon;
    private ExternalObserver eo;

    // Constructor
    public Model(String type, Integer[] structure, String[] functions, Integer epochs, double lr, String optimizer,
            double beta1,
            double beta2, double epsilon, Integer batchSize, double lambda, ExternalObserver eo)
            throws IllegalAccessException {

        this.structure = structure;
        this.lr = lr;
        this.opt = optimizer;
        this.functions = functions;
        this.type = type;
        this.activation = new Activation(structure, functions);
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.optimizer = new Optimizer(optimizer, structure, lr, beta1, beta2, epsilon);
        this.lambda = lambda;
        this.cost = new Loss(type);
        this.eo = eo;

        this.B = new double[structure.length][];
        this.W = new double[structure.length - 1][][];

        for (int i = 0; i < structure.length; i++) {
            this.B[i] = Initializer.initialize(structure[i], functions[i]);
        }

        for (int i = 0; i < structure.length - 1; i++) {
            this.W[i] = Initializer.initialize(structure[i], structure[i + 1], functions[i + 1]);
        }
    }

    // Train method
    public void train(double[][] X, double[][] Y, double[][] x, double[][] y) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < X.length; i++) {
                indices.add(i);
            }
            Collections.shuffle(indices);
            for (int start = 0; start < X.length; start += batchSize) {
                int end = Math.min(start + batchSize, X.length);

                double[][][] deltaW = new double[W.length][][];
                double[][] deltaB = new double[B.length][];

                for (int i = 0; i < W.length; i++) {
                    deltaW[i] = new double[W[i].length][];
                    for (int j = 0; j < W[i].length; j++) {
                        deltaW[i][j] = new double[W[i][j].length];
                    }
                }

                for (int i = 0; i < B.length; i++) {
                    deltaB[i] = new double[B[i].length];
                }

                for (int list_index = start; list_index < end; list_index++) {
                    int curr = indices.get(list_index);

                    double[][] Z = new double[structure.length][];
                    double[][] A = new double[structure.length][];

                    // Forward pass

                    for (int i = 0; i < structure.length; i++) {
                        Z[i] = new double[structure[i]];
                        A[i] = new double[structure[i]];
                    }

                    for (int i = 0; i < structure[0]; i++) {
                        Z[0][i] = X[curr][i];
                    }

                    A[0] = activation.activate(Z[0], 0);

                    for (int i = 0; i < structure.length - 1; i++) {
                        for (int j = 0; j < structure[i]; j++) {
                            for (int k = 0; k < structure[i + 1]; k++) {
                                Z[i + 1][k] += A[i][j] * W[i][j][k];
                            }
                        }

                        for (int j = 0; j < structure[i + 1]; j++) {
                            Z[i + 1][j] += B[i + 1][j];
                        }
                        A[i + 1] = activation.activate(Z[i + 1], i + 1);
                    }

                    // Back Propogation

                    double[] dCost = cost.derivative(Y[curr], A[structure.length - 1]);

                    double[] from = activation.differentiate(Z[structure.length - 1], structure.length - 1);

                    for (int i = 0; i < structure[structure.length - 1]; i++)
                        from[i] *= dCost[i];

                    for (int i = structure.length - 2; i >= 0; i--) {

                        double[] to = new double[structure[i]];
                        Arrays.fill(to, 0d);

                        for (int j = 0; j < structure[i]; j++) {
                            double diff[] = activation.differentiate(Z[i], i);
                            for (int k = 0; k < structure[i + 1]; k++) {
                                to[j] += diff[j] * from[k] * W[i][j][k];
                                deltaW[i][j][k] += from[k] * A[i][j];
                            }
                        }

                        for (int j = 0; j < structure[i + 1]; j++) {
                            deltaB[i + 1][j] += from[j];
                        }

                        from = to;
                    }
                }

                for (int i = 0; i < structure.length - 1; i++) {
                    for (int j = 0; j < structure[i]; j++) {
                        for (int k = 0; k < structure[i + 1]; k++) {
                            double dW = deltaW[i][j][k] / batchSize;
                            double l1Regularization = lambda * Math.signum(W[i][j][k]);
                            W[i][j][k] = optimizer.compute(W[i][j][k], dW + l1Regularization, i, j, k);
                        }
                    }
                }

                for (int i = 0; i < structure.length; i++) {
                    for (int j = 0; j < structure[i]; j++) {
                        double dB = deltaB[i][j] / batchSize;
                        double l1Regularization = lambda * Math.signum(B[i][j]);
                        B[i][j] = optimizer.compute(B[i][j], dB + l1Regularization, i, j);
                    }
                }
            }

            if (epoch % 100 == 0) {
                double[][] predicted = predict(x);
                System.out.println("\u001B[31m Epoch\u001B[36m [" + epoch + "] \u001B[34m R2 = \u001B[36m"
                        + calculateR2(y, predicted) + "\u001B[0m");
            }

            if(eo.ping(epoch)){
                eo.run(X);
            }
        }
    }

    public double calculateR2(double[][] actual, double[][] predicted) {
        return cost.calculateR2(actual, predicted);
    }

    public double[][] predict(double[][] x) {
        double[][] ret = new double[x.length][structure[structure.length - 1]];
        for (int curr = 0; curr < x.length; curr++) {
            double[][] Z = new double[structure.length][];
            double[][] A = new double[structure.length][];

            // Forward pass

            for (int i = 0; i < structure.length; i++) {
                Z[i] = new double[structure[i]];
                A[i] = new double[structure[i]];
            }

            for (int i = 0; i < structure[0]; i++) {
                Z[0][i] = x[curr][i];
            }

            A[0] = activation.activate(Z[0], 0);

            for (int i = 0; i < structure.length - 1; i++) {
                for (int j = 0; j < structure[i]; j++) {
                    for (int k = 0; k < structure[i + 1]; k++) {
                        Z[i + 1][k] += A[i][j] * W[i][j][k];
                    }
                }

                for (int j = 0; j < structure[i + 1]; j++) {
                    Z[i + 1][j] += B[i + 1][j];
                }

                A[i + 1] = activation.activate(Z[i + 1], i + 1);
            }

            for (int i = 0; i < structure[structure.length - 1]; i++) {
                ret[curr][i] = A[structure.length - 1][i];
            }
        }
        return ret;
    }

    public void export() {
        try (FileWriter writer = new FileWriter("wb.pt")) {
            writer.write("structure_length " + structure.length + " ");
            for (int i = 0; i < structure.length; i++)
                writer.write(structure[i] + " ");
            writer.write("\ntype " + type);
            writer.write("\nfunctions ");
            for (int i = 0; i < structure.length; i++)
                writer.write(functions[i] + " ");
            writer.write("\nepochs " + epochs);
            writer.write("\noptimizer " + opt);
            writer.write("\nlr " + lr);
            writer.write("\nbeta1 " + beta1);
            writer.write("\nbeta2 " + beta2);
            writer.write("\nepsilon " + epsilon);
            writer.write("\nlambda " + lambda);
            writer.write("\n\n");
            writer.write("weights" + "\n");
            for (int i = 0; i < W.length; i++) {
                for (int j = 0; j < W[i].length; j++) {
                    for (int k = 0; k < W[i][j].length; k++) {
                        writer.write(W[i][j][k] + " ");
                    }
                    writer.write("\n");
                }
                writer.write("\n");
            }
            writer.write("biases" + "\n");
            for (int i = 0; i < B.length; i++) {
                for (int j = 0; j < B[i].length; j++) {
                    writer.write(B[i][j] + " ");
                }
                writer.write("\n");
            }
            System.out.println("\u001B[31m Weights and biases exported to\u001B[1;31m wb.pt" + "\u001B[0m");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}