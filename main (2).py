from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserIconView
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.spinner import MDSpinner
import tensorflow as tf
import numpy as np
import pandas as pd

class PredictionApp(App):
    def build(self):
        self.title = "FRUITS & VEGETABLES RECOGNITION SYSTEM"
        self.orientation = "vertical"

        # Sidebar
        self.sidebar = Spinner(text='Home', values=('Home', 'About Project', 'Prediction'))
        self.sidebar.bind(text=self.on_sidebar_change)
        self.sidebar.size_hint_y = None
        self.sidebar.height = 40
        self.root_layout = BoxLayout(orientation='horizontal')
        self.root_layout.add_widget(self.sidebar)

        self.home_page()

        return self.root_layout

    def on_sidebar_change(self, spinner, text):
        self.root_layout.clear_widgets()
        if text == 'Home':
            self.home_page()
        elif text == 'About Project':
            self.about_page()
        elif text == 'Prediction':
            self.prediction_page()

    def home_page(self):
        image = Image(source="thomas-le-pRJhn4MbsMM-unsplash.jpg")
        self.root_layout.add_widget(image)

    def about_page(self):
        about_label = Label(text="About Project")
        dataset_info = Label(text="This dataset contains images of the following food items:\n\nfruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.\n\nvegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
        content_label = Label(text="This dataset contains three folders:\n\n1. train (100 images each)\n\n2. test (10 images each)\n\n3. validation (10 images each)")
        self.root_layout.add_widget(about_label)
        self.root_layout.add_widget(dataset_info)
        self.root_layout.add_widget(content_label)

    def prediction_page(self):
        prediction_label = Label(text="Model Prediction")
        self.file_chooser = FileChooserIconView(path='.')
        self.root_layout.add_widget(prediction_label)
        self.root_layout.add_widget(self.file_chooser)
        self.predict_button = Button(text="Predict")
        self.predict_button.bind(on_press=self.predict)
        self.root_layout.add_widget(self.predict_button)
        self.prediction_result = Label(text="")
        self.root_layout.add_widget(self.prediction_result)

    def predict(self, instance):
        if not self.file_chooser.selection:
            self.prediction_result.text = "Please select an image."
            return

        test_image_path = self.file_chooser.selection[0]
        # Load TensorFlow model
        model = tf.keras.models.load_model("trained_model.h5")
        image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(64, 64))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        predictions = model.predict(input_arr)
        predicted_index = np.argmax(predictions)
        # Read labels
        with open("labels.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        predicted_item = labels[predicted_index]
        # Fetch market prices
        market_price = self.get_market_price(predicted_item)
        self.prediction_result.text = f"Model is Predicting it's a {predicted_item}. Market price: {market_price} Rs"

    def get_market_price(self, predicted_item):
        excel_file = 'pricelist2.xlsx'
        df = pd.read_excel(excel_file)
        price = df.loc[df['vegatables and fruits '] == predicted_item, 'price'].values
        return price[0] if len(price) > 0 else "Not available"

if __name__ == "__main__":
    PredictionApp().run()
