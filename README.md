# Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project using a Recurrent Neural Network (RNN) to classify IMDB movie reviews as positive or negative based on the review text. This end-to-end pipeline includes data preprocessing, model building, evaluation, deployment using Gradio, and saving the model for future inference.

## Project Features
📥 Loads and preprocesses the IMDB dataset from a ZIP file

🧹 Text preprocessing with tokenization and padding

🧠 Built with a Simple RNN using TensorFlow/Keras

🧪 Splits data into training and testing sets

📊 Tracks accuracy and loss on validation and test data

💾 Saves trained model (model.h5) and tokenizer (tokenizer.pkl)

🧪 Predictive function for user input review

🌐 Deploys a Gradio-based web app for real-time predictions

## Dataset
- Source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- review: Movie review (text)
- sentiment: Label (positive or negative)

## Working

The project begins with data extraction by unzipping and loading the IMDB dataset, which contains movie reviews along with their corresponding sentiments. In the preprocessing phase, the sentiment labels are converted into binary format where “positive” is mapped to 1 and “negative” to 0. The review texts are then tokenized using a Tokenizer, and all sequences are padded to a fixed length of 200 to ensure uniform input shape for the model.

The neural network model is built using TensorFlow and consists of an Embedding layer that transforms words into dense vectors, followed by a Simple RNN layer to capture sequential patterns in the text, and a final Dense layer with a sigmoid activation function to predict binary sentiment output.

For training, the dataset is split into 80% training data and 20% test data. The model is trained for 5 epochs with a batch size of 64. After training, it is evaluated on the test dataset to compute the final loss and accuracy.

Once the model is trained and evaluated, it is saved as model.h5 and the tokenizer is saved using joblib as tokenizer.pkl for future inference. Finally, a Gradio interface is used to deploy a simple web application, allowing users to input any movie review and get an instant sentiment prediction (positive or negative).

## Output
<img width="1902" height="291" alt="image" src="https://github.com/user-attachments/assets/22aeb10c-930b-453b-8033-57b68fb827dc" />
<img width="1902" height="296" alt="image" src="https://github.com/user-attachments/assets/436b6156-fd1a-4a74-9847-34b906f78556" />

Link:
https://556a3208384d20d753.gradio.live/
