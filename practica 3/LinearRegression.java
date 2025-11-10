import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;


public class LinearRegression {

    private double[] weights; // w1, w2, ..., wm
    private double bias;      // término independiente b

    public LinearRegression() {
        this.bias = 0;
    }

    // --------------------------
    // Escalado Min-Max (por columna)
    // --------------------------
    public double[][] scaleData(double[][] X) {
        int rows = X.length;
        int cols = X[0].length;
        double[][] scaled = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            double min = X[0][j];
            double max = X[0][j];
            for (int i = 0; i < rows; i++) {
                if (X[i][j] < min) min = X[i][j];
                if (X[i][j] > max) max = X[i][j];
            }
            double denom = max - min;
            if (denom == 0) denom = 1e-9; // evita división por cero si constante
            for (int i = 0; i < rows; i++) {
                scaled[i][j] = (X[i][j] - min) / denom;
            }
        }
        return scaled;
    }

    // --------------------------
    // Fit: usa la ecuación normal:
    // w = (X^T X)^-1 X^T y
    // --------------------------
    public void fit(double[][] X, double[] y) {
        int n = X.length;
        int m = X[0].length;

        // Agregar columna de 1's (bias) -> Xb: n x (m+1)
        double[][] Xb = new double[n][m + 1];
        for (int i = 0; i < n; i++) {
            Xb[i][0] = 1.0;
            for (int j = 0; j < m; j++) Xb[i][j + 1] = X[i][j];
        }

        double[][] Xt = transpose(Xb);
        double[][] XtX = multiply(Xt, Xb);
        double[][] XtY = multiply(Xt, to2D(y));

        // Intentamos invertir XtX; si falla, avisamos.
        double[][] XtX_inv = null;
        try {
            XtX_inv = inverse(XtX);
        } catch (RuntimeException e) {
            System.out.println("Advertencia: la matriz (X^T X) parece singular y no es invertible.");
            throw e;
        }

        double[][] W = multiply(XtX_inv, XtY);

        // Guardar bias y pesos
        this.bias = W[0][0];
        this.weights = new double[m];
        for (int i = 0; i < m; i++) {
            this.weights[i] = W[i + 1][0];
        }
    }

    // --------------------------
    // Predict
    // --------------------------
    public double[] predict(double[][] X) {
        int n = X.length;
        double[] yPred = new double[n];
        for (int i = 0; i < n; i++) {
            double s = bias;
            for (int j = 0; j < X[0].length; j++) s += X[i][j] * weights[j];
            yPred[i] = s;
        }
        return yPred;
    }

    // --------------------------
    // Score: R^2 (coef. determinación)
    // --------------------------
    public double score(double[] yTrue, double[] yPred) {
        double mean = Arrays.stream(yTrue).average().orElse(0);
        double ssTot = 0.0, ssRes = 0.0;
        for (int i = 0; i < yTrue.length; i++) {
            ssTot += Math.pow(yTrue[i] - mean, 2);
            ssRes += Math.pow(yTrue[i] - yPred[i], 2);
        }
        if (ssTot == 0) return 0;
        return 1 - (ssRes / ssTot);
    }

    // --------------------------
    // Métodos auxiliares de álgebra matricial
    // --------------------------
    private double[][] transpose(double[][] A) {
        double[][] T = new double[A[0].length][A.length];
        for (int i = 0; i < A.length; i++)
            for (int j = 0; j < A[0].length; j++)
                T[j][i] = A[i][j];
        return T;
    }

    private double[][] multiply(double[][] A, double[][] B) {
        int r = A.length;
        int c = B[0].length;
        int inner = B.length;
        double[][] R = new double[r][c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                for (int k = 0; k < inner; k++)
                    R[i][j] += A[i][k] * B[k][j];
        return R;
    }

    // Inversión por Gauss-Jordan (modifica copia)
    private double[][] inverse(double[][] input) {
        int n = input.length;
        // Copia A (no modificar original)
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) System.arraycopy(input[i], 0, A[i], 0, n);

        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) I[i][i] = 1.0;

        for (int i = 0; i < n; i++) {
            // Pivot
            double pivot = A[i][i];
            if (Math.abs(pivot) < 1e-12) {
                // Buscar fila con valor no nulo y hacer swap
                int swapRow = -1;
                for (int r = i + 1; r < n; r++) {
                    if (Math.abs(A[r][i]) > 1e-12) { swapRow = r; break; }
                }
                if (swapRow == -1) throw new RuntimeException("Matriz singular, no se puede invertir.");
                double[] tmpA = A[i]; A[i] = A[swapRow]; A[swapRow] = tmpA;
                double[] tmpI = I[i]; I[i] = I[swapRow]; I[swapRow] = tmpI;
                pivot = A[i][i];
            }

            // Normalizar fila i
            for (int j = 0; j < n; j++) { A[i][j] /= pivot; I[i][j] /= pivot; }

            // Hacer ceros en otras filas
            for (int r = 0; r < n; r++) {
                if (r == i) continue;
                double factor = A[r][i];
                for (int c = 0; c < n; c++) {
                    A[r][c] -= factor * A[i][c];
                    I[r][c] -= factor * I[i][c];
                }
            }
        }
        return I;
    }

    private double[][] to2D(double[] y) {
        double[][] Y = new double[y.length][1];
        for (int i = 0; i < y.length; i++) Y[i][0] = y[i];
        return Y;
    }

    // --------------------------
    // Getters
    // --------------------------
    public double[] getWeights() { return weights; }
    public double getBias() { return bias; }

    // --------------------------
    // Carga CSV robusta:
    // - Detecta si hay header (intenta parsear la primera línea; si falla se asume header)
    // - Retorna DataSet con X y y (última columna es target)
    // --------------------------
    public static class DataSet {
        public double[][] X;
        public double[] y;
    }

    public static DataSet loadCSV(String filename) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        // Leer todo primero (para saber dimensiones)
        java.util.ArrayList<String> rows = new java.util.ArrayList<>();
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) rows.add(line);
        }
        br.close();
        if (rows.size() == 0) throw new IOException("Archivo vacío: " + filename);

        // Detectar header: intentar parsear la primera línea como números
        String[] firstSplit = rows.get(0).split(",");
        boolean hasHeader = false;
        try {
            for (String part : firstSplit) Double.parseDouble(part.trim());
        } catch (NumberFormatException ex) {
            hasHeader = true;
        }

        int startIdx = hasHeader ? 1 : 0;
        int n = rows.size() - startIdx;
        String[] sample = rows.get(startIdx).split(",");
        int cols = sample.length;
        if (cols < 2) throw new IOException("Formato inválido en CSV: se requieren al menos 2 columnas.");

        double[][] X = new double[n][cols - 1];
        double[] y = new double[n];

        for (int i = 0; i < n; i++) {
            String[] parts = rows.get(i + startIdx).split(",");
            for (int j = 0; j < cols - 1; j++) X[i][j] = Double.parseDouble(parts[j].trim());
            y[i] = Double.parseDouble(parts[cols - 1].trim());
        }
        DataSet ds = new DataSet();
        ds.X = X; ds.y = y;
        return ds;
    }

    // --------------------------
    // División aleatoria de índices (trainRatio p.ej. 0.8 -> 80% train)
    // --------------------------
    public static int[][] splitIndices(int total, double trainRatio, long seed) {
        int trainSize = (int) (total * trainRatio);
        int[] indices = new int[total];
        for (int i = 0; i < total; i++) indices[i] = i;
        shuffle(indices, seed);
        int[] trainIdx = Arrays.copyOfRange(indices, 0, trainSize);
        int[] testIdx = Arrays.copyOfRange(indices, trainSize, total);
        return new int[][]{trainIdx, testIdx};
    }

    private static void shuffle(int[] array, long seed) {
        Random rnd = new Random(seed);
        for (int i = array.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = array[i]; array[i] = array[j]; array[j] = tmp;
        }
    }

    // --------------------------
    // MAIN: ejecuta ambos experimentos
    // --------------------------
    public static void main(String[] args) {
        try {
            // --------------
            // 1) REGRESIÓN SIMPLE: Ice cream selling data (1 feature -> sales ~ temperature)
            // --------------
            System.out.println("=== Experimento: Ice Cream (regresión simple) ===");
            DataSet ice = loadCSV("Ice_cream_selling_data.csv"); 
            // Escalamos 
            LinearRegression modelIce = new LinearRegression();
            double[][] X_ice_scaled = modelIce.scaleData(ice.X);

            // Split 80/20 con semilla fija (reproducible)
            int[][] idxIce = splitIndices(X_ice_scaled.length, 0.8, 42L);
            int[] trainI = idxIce[0], testI = idxIce[1];

            double[][] X_ice_train = new double[trainI.length][X_ice_scaled[0].length];
            double[] y_ice_train = new double[trainI.length];
            double[][] X_ice_test = new double[testI.length][X_ice_scaled[0].length];
            double[] y_ice_test = new double[testI.length];

            for (int i = 0; i < trainI.length; i++) {
                X_ice_train[i] = X_ice_scaled[trainI[i]];
                y_ice_train[i] = ice.y[trainI[i]];
            }
            for (int i = 0; i < testI.length; i++) {
                X_ice_test[i] = X_ice_scaled[testI[i]];
                y_ice_test[i] = ice.y[testI[i]];
            }

            modelIce.fit(X_ice_train, y_ice_train);
            double[] yIcePred = modelIce.predict(X_ice_test);

            System.out.println("Pesos (ice): " + Arrays.toString(modelIce.getWeights()));
            System.out.println("Bias  (ice): " + modelIce.getBias());
            System.out.printf("R² (ice, test): %.4f%n", modelIce.score(y_ice_test, yIcePred));
            System.out.println();

            // --------------
            // 2) REGRESIÓN MÚLTIPLE: student_exam_scores.csv
            // --------------
            System.out.println("=== Experimento: Student Exam Scores (regresión múltiple) ===");
            DataSet stu = loadCSV("student_exam_scores.csv");
            LinearRegression modelStu = new LinearRegression();
            double[][] X_stu_scaled = modelStu.scaleData(stu.X);

            int[][] idxStu = splitIndices(X_stu_scaled.length, 0.8, 123L);
            int[] trainS = idxStu[0], testS = idxStu[1];

            double[][] X_stu_train = new double[trainS.length][X_stu_scaled[0].length];
            double[] y_stu_train = new double[trainS.length];
            double[][] X_stu_test = new double[testS.length][X_stu_scaled[0].length];
            double[] y_stu_test = new double[testS.length];

            for (int i = 0; i < trainS.length; i++) {
                X_stu_train[i] = X_stu_scaled[trainS[i]];
                y_stu_train[i] = stu.y[trainS[i]];
            }
            for (int i = 0; i < testS.length; i++) {
                X_stu_test[i] = X_stu_scaled[testS[i]];
                y_stu_test[i] = stu.y[testS[i]];
            }

            modelStu.fit(X_stu_train, y_stu_train);
            double[] yStuPred = modelStu.predict(X_stu_test);

            System.out.println("Pesos (stu): " + Arrays.toString(modelStu.getWeights()));
            System.out.println("Bias  (stu): " + modelStu.getBias());
            System.out.printf("R² (stu, test): %.4f%n", modelStu.score(y_stu_test, yStuPred));
            System.out.println();

            // Imprimir algunas predicciones vs real (ejemplo)
            System.out.println("Comparación (primeras 5 muestras del test):");
            int limit = Math.min(5, y_stu_test.length);
            for (int i = 0; i < limit; i++) {
                System.out.printf("real: %.3f, pred: %.3f%n", y_stu_test[i], yStuPred[i]);
            }

            System.out.println("\nFin de la ejecución. Asegúrate de tener los CSV en la misma carpeta que este .java");

        } catch (IOException e) {
            System.out.println("Error leyendo archivo CSV: " + e.getMessage());
        } catch (RuntimeException re) {
            System.out.println("Error de cálculo: " + re.getMessage());
        }
    }
}

