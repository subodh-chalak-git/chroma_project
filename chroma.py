import json
import os
import openai
import streamlit as st

with st.sidebar:
    st.title("Budget Speech Query")
    st.write("This app allows you to query the budget speech documents stored in ChromaDB.")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

if not OPENAI_API_KEY:
    st.error("Please provide a valid OpenAI API Key to proceed.")
    st.stop()

def extract_names(text: str) -> list[str]:
    system_prompt = "You are a name extractor. The user will give you text, and you must return a JSON array of names mentioned in the text. Do not include any explanation or formatting."

    response = openai.ChatCompletion.create(
        api_key=OPENAI_API_KEY,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )

    response = response.choices[0].message["content"]
    return json.loads(response)

text = "List the names of people in the following paragraph, separated by commas: Now the other princes of the Achaeans slept soundly the whole night through, but Agamemnon son of Atreus was troubled, so that he could get no rest. As when fair Hera's lord flashes his lightning in token of great rain or hail or snow when the snow-flakes whiten the ground, or again as a sign that he will open the wide jaws of hungry war, even so did Agamemnon heave many a heavy sigh, for his soul trembled within him. When he looked upon the plain of Troy he marveled at the many watchfires burning in front of Ilion... - The Iliad, Scroll 10"
st.write(extract_names(text))