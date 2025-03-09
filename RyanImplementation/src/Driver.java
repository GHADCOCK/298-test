import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import javafx.util.Duration;

import java.io.File;
import java.io.FileInputStream;
import java.util.*;

public class Driver extends Application {
    // Directory where the images are pulled
    public final static File[] IMAGE_FILES = new File("images").listFiles();
    // Size of the app. Number represents pixel length of one tile
    public static final int SIZE = 128;
    // If SIM is false, runs Q planning and saves the results. If SIM is true, runs a simulation of the cars
    private static final boolean SIM = true;

    public static final ImageView[] CARS = new ImageView[] {
            getImageView("redCar.png"),
            getImageView("yellowCar.png"),
            getImageView("blueCar.png")};
    public static State state;

    public static void main(String[] args) {
        if (!SIM) {
            MarkovGame.qPlanning();
            // MarkovGame.saveStatesAsObject("nash_equilibrium.ser");
            MarkovGame.saveStatesAsObject("nash_equilibrium.csv");
        } else {
            // MarkovGame.loadStatesAsObject("nash_equilibrium.ser");
            MarkovGame.loadStatesAsObject("nash_equilibrium.csv");
            state = MarkovGame.states[State.getStateIndex(
                    new int[] {1, 1, 0, 0, MarkovGame.WIDTH - 1, MarkovGame.HEIGHT - 1})];
            Application.launch(args);
        }
    }

    @Override
    public void start(Stage window) {
        Pane root = new Pane();

        createBoard(root);

        // Create a timeline that updates every second
        Timeline timeline = new Timeline(new KeyFrame(Duration.seconds(1), event -> {
            ActionSpace[] actions = new ActionSpace[MarkovGame.NUM_AGENTS];

            for (int a = 0; a < MarkovGame.NUM_AGENTS; a++) {
                actions[a] = MarkovGame.getMoveFromPolicy(state.NE[a]);
                switch (actions[a]) {
                    case UP -> CARS[a].setY(CARS[a].getY() - SIZE);
                    case DOWN -> CARS[a].setY(CARS[a].getY() + SIZE);
                    case LEFT -> CARS[a].setX(CARS[a].getX() - SIZE);
                    case RIGHT -> CARS[a].setX(CARS[a].getX() + SIZE);
                }
            }

            state = state.transition(actions);
        }));

        // Run timeline indefinitely
        timeline.setCycleCount(Timeline.INDEFINITE);
        timeline.play();

        Scene scene = new Scene(root, MarkovGame.WIDTH * SIZE, MarkovGame.HEIGHT * SIZE);

        window.setTitle("Car Game");
        window.setScene(scene);
        window.show();
    }

    private static ImageView getImageView(String spriteName) {
        try {
            for (File imageFile : Driver.IMAGE_FILES)
                if (imageFile.getName().equals(spriteName)) {
                    FileInputStream inStream = new FileInputStream(imageFile);
                    return new ImageView(new Image(inStream));
                }
        } catch (Exception e) {
            System.out.println(e.getClass() + ": " + e.getMessage());
        }

        throw new NoSuchElementException("Sprite not initialized");
    }

    private void createBoard(Pane root) {
        Group boardTiles = new Group();

        // Creates and draws the checkered grid based off of HEIGHT and WIDTH
        for (int y = 0; y < MarkovGame.HEIGHT; y++)
            for (int x = 0; x < MarkovGame.WIDTH; x++) {
                Rectangle tile = new Rectangle(x * SIZE, y * SIZE, SIZE, SIZE);

                tile.setFill((x + y) % 2 == 0 ? Color.BISQUE : Color.SADDLEBROWN);
                tile.setStroke(Color.BLACK);
                tile.setStrokeWidth(1);

                boardTiles.getChildren().add(tile);
            }

        root.getChildren().add(boardTiles);

        // Draws the cars based off the state
        for (int a = 0; a < MarkovGame.NUM_AGENTS; a++) {
            CARS[a].setX(state.state[2*a] * SIZE);
            CARS[a].setY(state.state[2*a+1] * SIZE);
            CARS[a].setFitWidth(SIZE);
            CARS[a].setFitHeight(SIZE);
            root.getChildren().add(CARS[a]);
        }
    }
}
