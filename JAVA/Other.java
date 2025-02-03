package JAVA;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class Other {
    public static void write(double[][] x , double[][] y , String path){
        try (FileWriter writer = new FileWriter(path)) {
            for (int i = 0; i < x.length; i++) {
                writer.write(Arrays.toString(x[i]) + " " + Arrays.toString(y[i]) + "\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void runPy(String[] args){
        try {

            ProcessBuilder processBuilder = new ProcessBuilder(args);

            // Start the process
            Process process = processBuilder.start();

            // Read the output of the Python script
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            // Wait for the Python script to finish
            process.waitFor();

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void deleteAll(File file) throws IOException {
        if(file.isDirectory()) {
            File[] files = file.listFiles();
            if (files != null) {
                for (File f : files) {
                    deleteAll(f);
                }
            }
        }else if(!file.delete()) {
            throw new IOException("Failed to delete " + file);
        }
    }
}
