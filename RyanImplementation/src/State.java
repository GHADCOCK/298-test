public class State {
    public static State[] getAllStates() {
        State[] states = new State[(int)(Math.pow(MarkovGame.WIDTH, MarkovGame.NUM_AGENTS) *
                Math.pow(MarkovGame.HEIGHT, MarkovGame.NUM_AGENTS))];

        for (int i = 0; i < states.length; i++)
            states[i] = indexToState(i);

        return states;
    }

    public static State indexToState(int index) {
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


    // Represented as (car 1's x, car 1's y, car 2's x, car 2's y)
    public final int[] state;
    public double[][] NE;
    private boolean offBoard;

    public double[] reward;
    private double[][][] QTable;
    private int[][] pastActions;
    private int[][] firstActions;


    // Used for Q planning
    public State (int[] state, boolean offBoard) {
        this.state = state;
        this.offBoard = offBoard;
        calculateReward();
    }

    // Used for simulation
    public State (int[] state, double[][] NE) {
        this.state = state;
        this.NE = NE;
    }


    public State transition(ActionSpace[] A) {
        int[] newState = state.clone();
        for (int i = 0; i < A.length; i++)
            switch (A[i]) {
                case LEFT -> newState[2*i]--;
                case RIGHT -> newState[2*i]++;
                case UP -> newState[2*i+1]--;
                case DOWN -> newState[2*i+1]++;
            }

        // Off the board, so not a part of states
        for (int i = 0; i < MarkovGame.NUM_AGENTS; i++)
            if (newState[2*i] < 0 || newState[2*i] >= MarkovGame.WIDTH ||
                    newState[2*i + 1] < 0 || newState[2*i + 1] >= MarkovGame.HEIGHT)
                return new State(newState, true);

        return MarkovGame.states[getStateIndex(newState)];
    }

    public void calculateReward() {
        reward = new double[MarkovGame.NUM_AGENTS];
        boolean[] isOffMap = new boolean[MarkovGame.NUM_AGENTS];
        int[] centerOfMap = new int[] {MarkovGame.WIDTH/2, MarkovGame.HEIGHT/2};

        for (int i = 0; i < reward.length; i++) {
            // Base tile reward (Calculated using the Manhattan distance from the center)
            reward[0] += (i == 0 ? 1 : -1) * (10 - 5 * (Math.abs(centerOfMap[0] - state[2*i]) +
                    Math.abs(centerOfMap[1] - state[2*i+1])));

            if (state[2*i] < 0 || state[2*i] >= MarkovGame.WIDTH ||
                state[2*i+1] < 0 || state[2*i+1] >= MarkovGame.HEIGHT)
                isOffMap[i] = true;
        }

        // Crash punishment
        if (state[0] == state[2] && state[1] == state[3])
            reward[0] -= 20;

        // 0 sum game
        reward[1] = -reward[0];

        // Off map punishment
        if (isOffMap[0])
            reward[1] = 1000;
        if (isOffMap[1])
            reward[0] = 1000;
        if (isOffMap[0])
            reward[0] = -1000;
        if (isOffMap[1])
            reward[1] = -1000;
    }

    public void initializeForFictitiousPlay() {
        QTable = new double[ActionSpace.values().length][ActionSpace.values().length][MarkovGame.NUM_AGENTS];
        for (int i = 0; i < ActionSpace.values().length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++) {
                State nextState = transition(new ActionSpace[] { ActionSpace.values()[i], ActionSpace.values()[j] });
                nextState.calculateReward();
                QTable[i][j] = nextState.reward.clone();
            }

        pastActions = new int[MarkovGame.NUM_AGENTS][ActionSpace.values().length];
        pastActions[0][0]++;
        pastActions[1][0]++;
    }

    public void updatePastActions() {
        ActionSpace[] A = bestResponse();

        pastActions[0][A[0].ordinal()]++;
        pastActions[1][A[1].ordinal()]++;
    }

    public void initializeFirstActions() {
        firstActions = new int[pastActions.length][ActionSpace.values().length];
        for (int i = 0; i < pastActions.length; i++)
            firstActions[i] = pastActions[i].clone();
    }

    public void forgetFirstActions() {
        for (int i = 0; i < pastActions.length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++)
                pastActions[i][j] -= firstActions[i][j];
    }

    // Returns the percentage the action has been taken by the agent
    private double getPercentAction(int agent, int action) {
        return pastActions[agent][action] / (double) MarkovGame.numEpisodes;
    }

    public void updateQTable() {
        for (int i = 0; i < ActionSpace.values().length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++) {
                State nextState = transition(new ActionSpace[] {ActionSpace.values()[i], ActionSpace.values()[j]});
                QTable[i][j] = nextState.expectedValues();
            }
    }

    public ActionSpace[] bestResponse() {
        ActionSpace[] bestActions = new ActionSpace[MarkovGame.NUM_AGENTS];

        double bestReward = Integer.MIN_VALUE;
        double currentReward;
        for (int i = 0; i < ActionSpace.values().length; i++) {
            currentReward = 0;
            for (int j = 0; j < ActionSpace.values().length; j++)
                currentReward += QTable[i][j][0] * getPercentAction(1, j);

            if (currentReward > bestReward) {
                bestActions[0] = ActionSpace.values()[i];
                bestReward = currentReward;
            }
        }

        bestReward = Integer.MIN_VALUE;
        for (int j = 0; j < ActionSpace.values().length; j++) {
            currentReward = 0;
            for (int i = 0; i < ActionSpace.values().length; i++)
                currentReward += QTable[i][j][1] * getPercentAction(0, i);

            if (currentReward > bestReward) {
                bestActions[1] = ActionSpace.values()[j];
                bestReward = currentReward;
            }
        }

        return bestActions;
    }

    public double[] expectedValues() {
        // Doesn't make sense to consider expected values of tiles off the board
        if (offBoard) {
            calculateReward();
            return reward.clone();
        }

        // Currently doesn't calculate the NE, just uses the fictitious play info
        double[] expVals = reward.clone();
        for (int i = 0; i < ActionSpace.values().length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++) {
                expVals[0] += MarkovGame.DISCOUNT_FACTOR * QTable[i][j][0] * getPercentAction(0, i) *
                        getPercentAction(1, j);
                expVals[1] += MarkovGame.DISCOUNT_FACTOR * QTable[i][j][1] * getPercentAction(0, i) *
                        getPercentAction(1, j);
            }

        return expVals;
    }

    public void calculateNE() {
        NE = new double[MarkovGame.NUM_AGENTS][ActionSpace.values().length];
        for (int i = 0; i < NE.length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++)
                NE[i][j] = getPercentAction(i, j);
    }

    // Code to calculate pure Nash if needed
    /*private List<double[][]> pureNash() {
        List<double[][]> pureNashEquilibiums = new ArrayList<>();

        double[] maxA1 = new double[ActionSpace.values().length];
        double[] maxA2 = new double[ActionSpace.values().length];

        List<double[][]> rewardMatrices = getRewardMatrices();

        for (int i = 0; i < ActionSpace.values().length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++) {
                if (rewardMatrices.get(0)[i][j] > maxA1[i])
                    maxA1[i] = rewardMatrices.get(0)[i][j];
                if (rewardMatrices.get(1)[i][j] > maxA2[j])
                    maxA2[j] = rewardMatrices.get(1)[i][j];
            }

        for (int i = 0; i < ActionSpace.values().length; i++)
            for (int j = 0; j < ActionSpace.values().length; j++)
                if (rewardMatrices.get(0)[i][j] == maxA1[i] && rewardMatrices.get(1)[i][j] == maxA2[j]) {
                    double[][] NE = new double[2][ActionSpace.values().length];
                    NE[0][i] = 1;
                    NE[1][j] = 1;
                    pureNashEquilibiums.add(NE);
                }

        return pureNashEquilibiums;
    }*/
}
