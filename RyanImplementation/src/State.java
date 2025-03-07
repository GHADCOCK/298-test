import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class State implements Serializable {
    private static final long serialVersionUID = 1L;

    public static State[] getAllStates() {
        State[] states = new State[(int)(Math.pow(MarkovGame.WIDTH, MarkovGame.NUM_AGENTS) *
                Math.pow(MarkovGame.HEIGHT, MarkovGame.NUM_AGENTS))];

        for (int i = 0; i < states.length; i++)
            states[i] = indexToState(i);

        return states;
    }

    private static State indexToState(int index) {
        int[] s = new int[2*MarkovGame.NUM_AGENTS];

        for (int i = 0; i < s.length; i++) {
            if (i % 2 == 0) {
                s[i] = index % MarkovGame.WIDTH;
                index /= MarkovGame.WIDTH;
            }
            else {
                s[i] = index % MarkovGame.HEIGHT;
                index /= MarkovGame.HEIGHT;
            }
        }

        return new State(s, false);
    }

    public static int getStateIndex(int[] state) {
        int index = 0;
        int mult = 1;

        for (int i = 0; i < state.length; i++) {
            index += state[i] * mult;

            if (i % 2 == 0)
                mult *= MarkovGame.WIDTH;
            else
                mult *= MarkovGame.HEIGHT;
        }

        return index;
    }


    // Represented as (car 1's x, car 1's y, car 2's x, car 2's y, ...)
    public final int[] state;
    // If there is a car which is off the board or not
    private final boolean offBoard;
    // The Nash equilibrium at this state
    public double[][] NE;

    public double[] reward;
    private Map<ActionSpace[], double[]> QTable;
    private double[] expectedValues;

    private int[][] pastActions;
    private int[][] firstActions;


    // Used for Q planning
    public State (int[] state, boolean offBoard) {
        this.state = state;
        this.offBoard = offBoard;
        calculateReward();
        expectedValues = reward.clone();
    }

    // Used for simulation
    public State (int[] state, double[][] NE) {
        this.state = state;
        this.NE = NE;
        offBoard = false;
    }


    public State transition(ActionSpace[] actions) {
        int[] newState = state.clone();
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
            switch (actions[a]) {
                case LEFT -> newState[2*a]--;
                case RIGHT -> newState[2*a]++;
                case UP -> newState[2*a+1]--;
                case DOWN -> newState[2*a+1]++;
            }

        // Off the board, so not a part of states
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
            if (newState[2*a] < 0 || newState[2*a] >= MarkovGame.WIDTH ||
                    newState[2*a+1] < 0 || newState[2*a+1] >= MarkovGame.HEIGHT)
                return new State(newState, true);

        return MarkovGame.states[getStateIndex(newState)];
    }

    public void calculateReward() {
        reward = new double[MarkovGame.NUM_AGENTS];

        // Off map punishment (If both run off the map, the rewards don't sum up to 0, but okay since neither will take
        // this action)
        /*if (isOffMap[0] && isOffMap[1]){
            reward[0] = -1000;
            reward[1] = -1000;
        }
        else if (isOffMap[0]) {
            reward[0] = -1000;
            reward[1] = 1000;
        }
        else if (isOffMap[1]) {
            reward[0] = 1000;
            reward[1] = -1000;
        }*/
        if (offBoard) {
            boolean[] isOffMap = new boolean[MarkovGame.NUM_AGENTS];
            for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
                if (state[2*a] < 0 || state[2*a] >= MarkovGame.WIDTH ||
                        state[2*a+1] < 0 || state[2*a+1] >= MarkovGame.HEIGHT)
                    isOffMap[a] = true;

            // if (isOffMap[0] && isOffMap[1] && isOffMap[2]) {
            //     reward[0] = -2000;
            //     reward[1] = -1000;
            //     reward[2] = -1000;
            // } else if (isOffMap[0]) {
            //     reward[0] = -2000;
            //     reward[1] = 1000;
            //     reward[2] = 1000;
            // } else {
            //     // One of the evaders is off the map
            //     reward[0] = 2000;
            //     reward[1] = -1000;
            //     reward[2] = -1000;
            // }
            if (isOffMap[0] && isOffMap[1]){
                reward[0] = -1000;
                reward[1] = -1000;
            }
            else if (isOffMap[0]) {
                reward[0] = -1000;
                reward[1] = 1000;
            }
            else if (isOffMap[1]) {
                reward[0] = 1000;
                reward[1] = -1000;
            }
            return;
        }

        int[] centerOfMap = new int[]{MarkovGame.WIDTH / 2, MarkovGame.HEIGHT / 2};

        // Calculate evader's reward
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++) {
            int pursuerMult = a == 0 ? -1 : 1;
            // Base tile reward for first agent (Calculated using the Manhattan distance from the center)
            reward[1] += pursuerMult * (10 - 5 * (Math.abs(centerOfMap[0] - state[2*a]) +
                    Math.abs(centerOfMap[1] - state[2*a+1])));

            // Crash punishment for first agent
            for (int i = a + 1; i < reward.length; i++)
                if (i != a && state[2*a] == state[2*i] && state[2*a+1] == state[2*i+1])
                    reward[1] -= 40;
        }

        // reward[2] = reward[1];

        // Make reward for second agent so that it's a 0 sum game
        // reward[0] = -2 * reward[1];
        reward[0] = -reward[1];
    }

    public void initializeForFictitiousPlay() {
        QTable = new HashMap<>();
        for (ActionSpace[] each : ActionSpace.ALL_ACTIONS) {
            State nextState = transition(each);
            QTable.put(each, nextState.reward.clone());
        }

        pastActions = new int[MarkovGame.NUM_AGENTS][ActionSpace.values().length];
        // Performs up as the first move at every state
        for (int[] each : pastActions)
            for (int i = 0; i < ActionSpace.values().length; i++)
                each[i]++;
    }

    public void updatePastActions() {
        ActionSpace[] actions = bestResponse();
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
            pastActions[a][actions[a].ordinal()]++;
    }

    public void initializeFirstActions() {
        firstActions = new int[MarkovGame.NUM_AGENTS][ActionSpace.values().length];
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
            firstActions[a] = pastActions[a].clone();
    }

    public void forgetFirstActions() {
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
            for (int i = 0; i < ActionSpace.values().length; i++)
                pastActions[a][i] -= firstActions[a][i];
    }

    // Returns the percentage the action has been taken by the agent
    private double getPercentAction(int agent, int action) {
        return pastActions[agent][action] / (double) MarkovGame.numEpisodes;
    }

    public void updateQTable() {
        for (ActionSpace[] each : ActionSpace.ALL_ACTIONS) {
            State nextState = transition(each);
            QTable.replace(each, nextState.expectedValues.clone());
        }
    }

    public ActionSpace[] bestResponse() {
        double[][] expectedValues = new double[MarkovGame.NUM_AGENTS][ActionSpace.values().length];

        for (ActionSpace[] each : ActionSpace.ALL_ACTIONS) {
            double[] QValues = QTable.get(each);
            double mult = 1;

            for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
                mult *= getPercentAction(a, each[a].ordinal());

            for (int a = 0; a < MarkovGame.NUM_AGENTS; a++) {
                int currAgentAction = each[a].ordinal();
                expectedValues[a][currAgentAction] += QValues[a] * mult / getPercentAction(a, currAgentAction);
            }
        }

        ActionSpace[] bestResponses = new ActionSpace[MarkovGame.NUM_AGENTS];
        double maxVal;

        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++) {
            maxVal = Integer.MIN_VALUE;
            for (int i = 0; i < ActionSpace.values().length; i++)
                if (expectedValues[a][i] > maxVal) {
                    maxVal = expectedValues[a][i];
                    bestResponses[a] = ActionSpace.values()[i];
                }
        }

        return bestResponses;
    }

    public void updateExpectedValues() {
        expectedValues = reward.clone();
        double[] QValues;
        double mult;

        for (ActionSpace[] each : ActionSpace.ALL_ACTIONS) {
            QValues = QTable.get(each);
            mult = MarkovGame.DISCOUNT_FACTOR;

            for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
                mult *= getPercentAction(a, each[a].ordinal());

            for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
                expectedValues[a] += QValues[a] * mult;
        }
    }

    public void calculateNE() {
        NE = new double[MarkovGame.NUM_AGENTS][ActionSpace.values().length];
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++)
            for (int i = 0; i < ActionSpace.values().length; i++)
                NE[a][i] = getPercentAction(a, i);
    }
}
