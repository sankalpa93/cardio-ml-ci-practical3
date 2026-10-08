import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_file_created(self):
        self.assertTrue(os.path.exists("cardio_model.pkl"))

    def test_metrics_file_valid(self):
        with open("metrics.json") as file:
            metrics = json.load(file)
        self.assertIn("accuracy", metrics)
        self.assertGreater(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_can_predict(self):
        model = joblib.load("cardio_model.pkl")
        sample = pd.DataFrame([{
            "age_years": 50, "gender": 1, "height": 165, "weight": 70.0,
            "ap_hi": 120, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1,
        }])
        prediction = model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])


if __name__ == "__main__":
    unittest.main()
