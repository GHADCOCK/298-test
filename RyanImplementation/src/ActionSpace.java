public enum ActionSpace {
    UP,
    DOWN,
    LEFT,
    RIGHT,
    STOP;

    public static final ActionSpace[][] ALL_ACTIONS = calculateAllActions();

    private static ActionSpace[][] calculateAllActions() {
        ActionSpace[][] allActions = new ActionSpace[
                (int)Math.pow(ActionSpace.values().length, MarkovGame.NUM_AGENTS)][MarkovGame.NUM_AGENTS];

        for (int i = 0; i < allActions.length; i++)
            allActions[i] = indexToActions(i);

        return allActions;
    }

    private static ActionSpace[] indexToActions(int index) {
        ActionSpace[] actions = new ActionSpace[MarkovGame.NUM_AGENTS];

        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++) {
            actions[a] = ActionSpace.values()[index % values().length];
            index /= values().length;
        }

        return actions;
    }
}