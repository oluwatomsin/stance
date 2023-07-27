import streamlit as st
from streamlit_lottie import st_lottie
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification#, RobertaTokenizerFast, TFRobertaForSequenceClassification
import torch
import json



# Loading the model and caching it to prevent reload
# @st.cache(allow_output_mutation=True)
def load_model():
    """
    Load the models into memory
    Output:
        Tuple: First entry contains sentiment model, second contains tuple of stance model and tokenizer
    """

    # Instantiate stance detection model
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.use_multiprocessing = False

    # Sentiment analysis
    sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")

    return (sentiment_analysis, (model, tokenizer))


# Creating a function that gets the sentiment and stance of a model
def detect_bias(models, text):
  """
  Takes in health related text and outputs stance and sentiment
  Input:
      text (str): text data
      models (tuple): First model should be sentiment model, second should be stance model. This will ideally run on the output of the load model function
  Output:
      tuple of dictionaries: Returns a tuple where the first entry contains the sentiment dictionary of sentiment strength pairs and the second contains a dictionary of stance dictionary pairs.
  """
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  premise = text
  hypothesis = "The health concerns are valid"

  sent_model = models[0]
  stance_model = models[1][0]
  stance_tokenizer = models[1][1]

  #Stance prediction
  input = stance_tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
  output = stance_model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
  prediction_stance = torch.softmax(output["logits"][0], -1).tolist()
  label_names = ["entailment", "neutral", "contradiction"]
  prediction_stance = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction_stance, label_names)}

  return (sent_model(text)[0], prediction_stance)


# loading the model into memory
models = load_model()


# lotties link
"https://lottiefiles.com/search?q=members&category=animations"
# Creating function for loading lottie as a json file
def load_lottie_files(filepath: str):
    with open(file=filepath, mode='r') as f:
        return json.load(f)
    

loffie_sentiment = load_lottie_files(filepath="sent.json")
loffie_intro = load_lottie_files(filepath="sent_intro.json")
loffie_members = load_lottie_files(filepath="sent_members.json")


# Page 1: Introduction to Streamlit Project
def page_intro():
    """
    This function contains the information to be presented on the intro page.
    -----------------------------------------
    This contain information about what the project is about and what the deliverables.
    Information is alsoprovided about the porject partner[Learnroll]
    """
    st.title("Introduction to Streamlit Project")
    st.write("Welcome to our Streamlit application! This project aims to showcase the power of Streamlit for creating interactive web applications.")
    st_lottie(
        loffie_intro,
        speed=1,
        reverse=False,
        loop=True,
        quality='high',
        height=400
        #renderer="svg"
    )
    # Add any other introductory content here.

# Page 2: Stance and Sentiment Analysis
def page_analysis():
    """
    """
    st.title("Stance and Sentiment Analysis")
    st_lottie(
        loffie_sentiment,
        speed=1,
        reverse=False,
        loop=True,
        quality='high',
        height=400
        #renderer="svg"
    )
    user_input = st.text_area("Enter your text here:", "Type your text here...")
    if st.button("Analyze", type='primary'):
        # Perform stance and sentiment analysis on user_input
        # Implement your analysis code here and display the results
        st.write(f"Analysis: {detect_bias(models, user_input)}")
        # Display the analysis results here.

# Page 3: Information about Participants
def page_participants():
    st.title("Information about Participants")
    st.write("This page provides information about the participants in the Streamlit project.")
    st_lottie(
        loffie_members,
        speed=1,
        reverse=False,
        loop=True,
        quality='high',
        height=400
        #renderer="svg"
    )
    # Add information about the participants here.
    # You can use st.write, st.markdown, or any other Streamlit elements.

# Main function to run the Streamlit app
def main():
    st.snow()
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ("Introduction", "Stance and Sentiment Analysis", "Participants"))

    if page == "Introduction":
        page_intro()
    elif page == "Stance and Sentiment Analysis":
        page_analysis()
    elif page == "Participants":
        page_participants()

if __name__ == "__main__":
    main()
