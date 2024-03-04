from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import streamlit as st
from torch import cuda

if cuda.is_available():
    device='cuda'
else:
    device='cpu'

@st.cache_resource()

def load_model():
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model.to(device)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return model,tokenizer

model,tokenizer=load_model()
st.title("Multilingual translation app")

st.write("This app demonstrates translation capabilities of LLM.The app leverages M2M100_418M model by facebook")



col1,col2 = st.columns(2)

with col1:
    source_language=st.radio("Select source language",["ar","zh","de","bn","Kn","ta"])
    user_text = st.text_area("Enter text for translation")
   

with col2:
   target_language=st.radio("Select target language",["en","de","bn","hi","kn","ta"])
   if user_text:
        tokenizer.src_lang = source_language#"zh"#"hi"
        encoded_text = tokenizer(user_text, return_tensors="pt").to(device)
        generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(target_language))
        m2m_translated=tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        st.write(m2m_translated)
        #st.snow()
       





