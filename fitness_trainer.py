import os
from collections import deque

from DIPPID import SensorUDP
import pandas as pd
import pyglet

from activity_recognizer import (
    exercise_name_list,
    images_directory,
    samples_size_per_exercise,
    seconds_required_for_exercise,
    build_dataset,
    ExerciseClassifier,
)

window_width = 1920
window_height = 1080


def load_exercise_image(exercise_index):
    exercise_name = exercise_name_list[exercise_index]
    image_path = os.path.join(images_directory, f"{exercise_name}_2.png")
    return pyglet.image.load(image_path)


class TrainerWindow(pyglet.window.Window):
    def __init__(self, classifier, sensor):
        pyglet.gl.glClearColor(100 / 255, 150 / 255, 200 / 255, 1)
        super().__init__(window_width, window_height, "Fitness Trainer")

        sound_directory = os.path.join(os.path.dirname(__file__), "soundeffects")
        background_music_path = os.path.join(sound_directory, "background_track.mp3")
        self.background_player = pyglet.media.Player()
        self.background_player.queue(pyglet.media.load(background_music_path))
        self.background_player.loop = True
        self.background_player.play()
        self.success_sound = pyglet.media.load(os.path.join(sound_directory, "yes.ogg"), streaming=False)

        self.classifier = classifier
        self.sensor = sensor
        self.current_exercise_index = 0
        self.accelerometer_samples = deque(maxlen=samples_size_per_exercise)
        self.gyroscope_samples = deque(maxlen=samples_size_per_exercise)
        self.time_spent = 0.0

        self.batch = pyglet.graphics.Batch()

        self.background_rectangle = pyglet.shapes.Rectangle(
            0, 0, window_width, window_height, (100, 150, 200, 255), batch=self.batch
        )

        self.exercise_sprite = None
        self.change_exercise_sprite()

        self.goal_label = pyglet.text.Label(
            "", x=window_width // 2, y=window_height - 70,
            anchor_x="center", font_size=60, color=(250, 250, 250, 255), batch=self.batch
        )
        self.prediction_label = pyglet.text.Label(
            "", x=window_width // 2, y=30,
            anchor_x="center", font_size=48, color=(253, 32, 32, 255), batch=self.batch
        )

        bar_width = 500
        bar_height = 30
        bar_x = (window_width - bar_width) // 2
        bar_y = 120
        self.progress_background = pyglet.shapes.Rectangle(
            bar_x, bar_y, bar_width, bar_height, (100, 100, 100, 255), batch=self.batch
        )
        self.progress_foreground = pyglet.shapes.Rectangle(
            bar_x, bar_y, 0, bar_height, (32, 253, 58, 255), batch=self.batch
        )

        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    def change_exercise_sprite(self):
        if self.exercise_sprite:
            self.exercise_sprite.delete()
        image = load_exercise_image(self.current_exercise_index)
        self.exercise_sprite = pyglet.sprite.Sprite(
            image, x=window_width // 2 - 260, y=window_height // 2 - 250, batch=self.batch
        )
        self.exercise_sprite.anchor_x = image.width // 2
        self.exercise_sprite.anchor_y = image.height // 2
        self.exercise_sprite.scale = min((window_width * 0.6) / image.width, (window_height * 0.6) / image.height)

    def advance_exercise(self):
        self.success_sound.play()
        self.current_exercise_index = (self.current_exercise_index + 1) % len(exercise_name_list)
        self.time_spent = 0.0
        self.change_exercise_sprite()

    def update(self, delta_time):
        accelerometer_data = self.sensor.get_value("accelerometer")
        gyroscope_data = self.sensor.get_value("gyroscope")
        if accelerometer_data and gyroscope_data:
            self.accelerometer_samples.append(
                (accelerometer_data["x"], accelerometer_data["y"], accelerometer_data["z"])
            )
            self.gyroscope_samples.append(
                (gyroscope_data["x"], gyroscope_data["y"], gyroscope_data["z"])
            )

        if len(self.accelerometer_samples) == samples_size_per_exercise:
            window_data_frame = pd.DataFrame(
                {
                    "acc_x": [v[0] for v in self.accelerometer_samples],
                    "acc_y": [v[1] for v in self.accelerometer_samples],
                    "acc_z": [v[2] for v in self.accelerometer_samples],
                    "gyro_x": [v[0] for v in self.gyroscope_samples],
                    "gyro_y": [v[1] for v in self.gyroscope_samples],
                    "gyro_z": [v[2] for v in self.gyroscope_samples],
                }
            )
            predicted_label, predicted_probability = self.classifier.predict(window_data_frame)

            correct = (
                predicted_label == exercise_name_list[self.current_exercise_index] and predicted_probability >= 0.65
            )
            self.prediction_label.color = (32, 253, 58, 255) if correct else (253, 32, 32, 255)
            self.prediction_label.text = f"Detected: {predicted_label} ({int(predicted_probability * 100)}%)"

            if correct:
                self.time_spent += delta_time
            progress = min(self.time_spent / seconds_required_for_exercise, 1.0)
            self.progress_foreground.width = self.progress_background.width * progress
            self.goal_label.text = (
                f"Do: {exercise_name_list[self.current_exercise_index]} "
                f"({int(self.time_spent)}/{seconds_required_for_exercise}s)"
            )
            if self.time_spent >= seconds_required_for_exercise:
                self.advance_exercise()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_close(self):
        self.sensor.disconnect()
        pyglet.app.exit()


def main():
    data_frame = build_dataset("data")
    classifier = ExerciseClassifier()
    classifier.train(data_frame)
    sensor = SensorUDP(port=5700)
    TrainerWindow(classifier, sensor)
    pyglet.app.run()


if __name__ == "__main__":
    main()
