import streamlit as st
import re
import requests

import requests

def call_api(text: str):
    # Replace this URL with your API endpoint
    api_url = "http://localhost:8090/complete?query_inp=" + text
    headers = {"accept": "application/json"}
    response = requests.post(api_url, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Failed to fetch text from API"
def replace_consecutive_dots(input_text):
    # Define regular expression to match three or more consecutive dots
    pattern = r'\.{3,}'
    # Replace consecutive dots with the result of the API call
    return re.sub(pattern, call_api(input_text), input_text, 1)

def main():
    st.title("API Text Replacement")

    # Text input box
    input_text = st.text_input("Enter text:")

    # Check if input text contains three or more consecutive dots
    if "..." in input_text:
        input_text = replace_consecutive_dots(input_text)

    # Display the updated text
    st.write(input_text)

if __name__ == "__main__":
    main()
