import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import pandas as pd
import time

# Set page configuration
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🍥",
    layout="wide"
)

# Header and description
st.title("Anime Recommendation")
st.markdown("""
This application recommends anime based on your preferences. Just tell me what anime you enjoy,
and I'll suggest similar titles you might like!
""")

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a Large Language Model (LLM) to provide personalized anime recommendations 
    based on your preferences. The recommendations are retrieved from a vector database 
    containing information about thousands of anime titles.
    """)
    
    st.header("How to use")
    st.markdown("""
    1. Enter the name of an anime you enjoyed or describe what kind of anime you're looking for
    2. Click the "Get Recommendations" button
    3. Browse through the personalized recommendations
    """)

# Initialize the LLM model
@st.cache_resource
def load_model():
    return OllamaLLM(model="llama3.2:1b")

# Main input area
user_query = st.text_area("Tell me what anime you like or describe what you're looking for:", 
                         height=100, 
                         placeholder="Example: I love Attack on Titan and Death Note. I enjoy dark themes with psychological elements.")

# Create prompt template
template = """
You are an Otaku assistant specialized in anime and manga recommendations. 
Your task is to recommend similar anime based on the user's preferences.

Given these anime entries:
{context}

User query: {query}

Provide 3-5 recommendations in this format:
- **Title**: [name]
  - **Type**: [TV/Movie/OVA]
  - **Genre**: [genre]
  - **Why Recommended**: [brief explanation]

Focus only on anime recommendations and avoid any off-topic responses.
"""

# Process button
if st.button("Get Recommendations", type="primary"):
    if user_query:
        # Show a spinner while processing
        with st.spinner("Finding the perfect anime recommendations for you..."):
            # Load model
            model = load_model()
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | model
            
            # Start time for performance tracking
            start_time = time.time()
            
            # Debug information (collapsible)
            with st.expander("Debug Information", expanded=False):
                try:
                    retrieved_docs = retriever.invoke(user_query)
                    st.write(f"Retrieved {len(retrieved_docs)} documents")
                    st.json([{
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    } for doc in retrieved_docs])
                except Exception as e:
                    st.error(f"Error retrieving documents: {str(e)}")
            
            # Process and display results
            try:
                result = chain.invoke({"context": retrieved_docs, "query": user_query})
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Create a nice results display
                st.success(f"✨ Found recommendations in {processing_time:.2f} seconds!")
                
                # Display the recommendations in a nice format
                st.markdown("## Your Personalized Recommendations")
                st.markdown(result)
                
                # Add feedback options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("👍 Helpful"):
                        st.success("Thank you for your feedback!")
                with col2:
                    if st.button("👎 Not helpful"):
                        st.info("Thanks for letting us know. Try refining your query for better results.")
                with col3:
                    if st.button("🔄 Try Again"):
                        st.experimental_rerun()
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
    else:
        st.warning("Please enter your anime preferences first.")

# Footer
st.markdown("---")
st.markdown("*Powered by LangChain and Ollama LLMs*")