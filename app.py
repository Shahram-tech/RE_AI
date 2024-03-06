from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load Airbnb dataset
airbnb_data = pd.read_csv("Airbnb_Data.csv")

# Initialize OpenAI language model
llm = OpenAI(temperature=0, openai_api_key="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")
memory = ConversationBufferMemory(llm=llm, max_token_limit=100)

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
  return render_template('index.html')


@app.route('/data', methods=['POST'])
def get_data():
  data = request.get_json()
  user_input = data.get('data')

  try:
    # Example conversation handling
    if "price" in user_input.lower():
      # Retrieve average price from the dataset
      average_price = airbnb_data['log_price'].mean()
      response = f"The average price is ${average_price:.2f}"
    else:
      # If the input does not match any predefined query, use the language model
      conversation = ConversationChain(llm=llm, memory=memory)
      response = conversation.predict(input=user_input)
      memory.save_context({"input": user_input}, {"output": response})

    # Appending a message
    response += "As Real Estate AI How may I assist you further?"

    return jsonify({"response": True, "message": response})

  except Exception as e:
    print(e)
    error_message = f'Error: {str(e)}'
    return jsonify({"message": error_message, "response": False})


if __name__ == '__main__':
  app.run(host="0.0.0.0", port=8080)
