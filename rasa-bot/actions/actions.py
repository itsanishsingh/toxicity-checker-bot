from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List

import tensorflow as tf
import pickle


class ActionToxicCheck(Action):

    def name(self) -> str:

        return "action_check_toxicity"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        serial_path = ".../serialized_models/"
        vectorizer = pickle.loads(serial_path + "tfidf_vectorizer.pkl")
        model = tf.keras.models.load_model(serial_path + "toxic_comment_model.h5")

        user_message = tracker.latest_message["text"]

        vect_text = vectorizer.transform(user_message)
        prediction = model.predict([vect_text])
        is_toxic = prediction[0] > 0.5

        if is_toxic:
            return [
                {"event": "slot", "name": "is_toxic", "value": True},
                {"event": "intent", "name": "toxic", "confidence": 0.5},
            ]
        else:
            return [{"event": "slot", "name": "is_toxic", "value": False}]
