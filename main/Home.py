import streamlit as st


st.title("Resume parsing approach summary")
st.info("Please test the model implemented for this parsing task from the tabs on the left.")


st.markdown("There are following steps applied in the parsing of a resume in a pdf or a doc format:")

st.markdown("**OCR section detection** - based on positioning and spacing of text groups")
st.markdown("**Section classification** - based on regex with keywords. This list can be expanded to a much larger corpus of identifiers. In case of `gpt-4-turbo` the section classification of text blocks is enhanced by the LLM itself.")
st.markdown("**Prompting with few shot examples** - the LLMs are called with a prompt along with few shot examples to ensure the accuracy of the JSON response.")

st.markdown("**Accuracy** - The exact accuracy of the model couldn't be calculated due to limited time. One approach to test and calculate the accuracy is to generate dummy data for resume fields as JSON files and convert them into pdf and word documents and run them through these models. The accuracy on a single file can be the percentage of fields that are accurately extracted. An incomplete field or missing field would count as inaccurate. This would give us a measure for accuracy especially when testing on a large dataset of documents. Sections can also be assigned weights based on their importance when measuring accuracy.")
st.markdown("**Other languages** - Other languages couldn't be covered in this version but we can address other languages by adding a layer of a Seq2Seq generator.")
st.markdown("**Improvements** - Given the growing number of transformers and architectures in AI especially for natural language processing, we can do a variety of things to improve these models. The most significant of these is fine-tuning a pre-trained model to labeled data. Fine-tuning couldn't be tested for this exercise as only training a big dataset can have a significant improvement over these large models.")
st.markdown("**Vision models** - I see a big opportunity of improvement with the use of vision models beside LLMs. Vision models can extract sections and entities at a much faster speed and accuracy if trained on a large dataset. The LLM approach is generative and requires more resources to achieve a speed that a vision based classification model's performance. A combination of vision and LLMs for the parts where they are advantageous would be ideal for parsing.")
st.markdown("**RAG with Vector DB** - Developing an RAG model over a vector database can make the document entity extraction much more efficient, fast and even more accurate.")
