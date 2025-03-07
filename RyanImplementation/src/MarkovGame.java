import java.io.*;
import java.util.Random;

public class MarkovGame {
    // Hyperparameters for the game
    public static final int HEIGHT = 3;
    public static final int WIDTH = 3;
    public static final int NUM_AGENTS = 2;

    // Hyperparameters for fictitious play
    public static final double DISCOUNT_FACTOR = 0.9;
    private static final double PERCENT_FORGET = 0.1;
    private static final int MAX_EPISODES = 2000;
    public static int numEpisodes;
    private static final Random r = new Random();

    public static State[] states;

    public static void qPlanning() {
        states = State.getAllStates();

        fictitiousPlay();

        for (State each : states)
            each.calculateNE();
    }

    public static void fictitiousPlay() {
        for (State each : states)
            each.initializeForFictitiousPlay();

        for (numEpisodes = ActionSpace.values().length; numEpisodes < MAX_EPISODES; numEpisodes++) {
            for (State each : states) {
                each.updatePastActions();
                each.updateExpectedValues();
                each.updateQTable();
            }

            if (numEpisodes < MAX_EPISODES * PERCENT_FORGET && (numEpisodes+1) >= MAX_EPISODES * PERCENT_FORGET)
                for (State each : states)
                    each.initializeFirstActions();

            if (numEpisodes % 100 == 0)
                System.out.println(numEpisodes);
        }

        for (State each : states)
            each.forgetFirstActions();

        numEpisodes *= 1 - PERCENT_FORGET;
    }

    public static ActionSpace getMoveFromPolicy(double[] policy) {
        double choice = r.nextDouble();
        for (int i = 0; i < policy.length; i++) {
            choice -= policy[i];
            if (choice <= 0)
                return ActionSpace.values()[i];
        }

        throw new IllegalArgumentException("Policy doesn't add to at least 1");
    }

    public static void saveStatesAsObject(String filename) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
            oos.writeObject(states);
            System.out.println("NE successfully saved to file: " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to retrieve NE from a file
    @SuppressWarnings("unchecked")
    public static void loadStatesAsObject(String filename) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
            states = (State[]) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    // Method to save states to CSV
    public static void saveStatesToCSV(String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write("X_1,Y_1,X_2,Y_2,U1,D1,L1,R1,S1,U2,D2,L2,R2,S2");
            writer.newLine();

            for (State s : states) {
                StringBuilder line = new StringBuilder();

                // Append state values
                for (int value : s.state)
                    line.append(value).append(",");

                // Remove trailing comma and write to file
                line.replace(line.length() - 1, line.length(), ":");

                // Append NE values
                for (double[] row : s.NE)
                    for (double value : row)
                        line.append(value).append(",");

                // Remove trailing comma and write to file
                writer.write(line.substring(0, line.length() - 1));
                writer.newLine();
            }

            System.out.println("States successfully saved to " + filename);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to load states from CSV
    public static void loadStatesFromCSV(String filename) {
        states = new State[(int)(Math.pow(WIDTH, NUM_AGENTS) * Math.pow(HEIGHT, NUM_AGENTS))];

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line = reader.readLine(); // Skip header

            for (int k = 0; (line = reader.readLine()) != null; k++) {
                String[] values = line.split(",");

                // Extract state values
                int[] state = new int[2*NUM_AGENTS];
                for (int i = 0; i < state.length; i++)
                    state[i] = Integer.parseInt(values[i]);

                // Extract NE values
                double[][] NE = new double[NUM_AGENTS][ActionSpace.values().length];
                int index = state.length;
                for (int i = 0; i < NE.length; i++)
                    for (int j = 0; j < ActionSpace.values().length; j++)
                        NE[i][j] = Double.parseDouble(values[index++]);

                states[k] = new State(state, NE);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
