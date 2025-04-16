# Anime Recommendation System

A personalized anime recommendation system powered by LangChain, Ollama, and vector search.(This is very Simple Implementation without complex recommendation algorithm or search.)
***NOTE : The retriever retrieves the most relevant information from the database and the LLM use that retreieved dataset as the base to give suggestion from the general perspective.***

#### Database Used : 
https://www.kaggle.com/datasets/muhammadishaque/anime-dataset-for-nlp

## Overview

This project creates a Streamlit web application that provides personalized anime recommendations based on user preferences. The system combines:

1. **Vector database search** - To find similar anime based on user input
2. **LLM recommendations** - To process and explain the results in natural language

The application uses a vector database created from anime metadata (stored in CSV format) and leverages the Llama 3.2 model via Ollama to generate personalized recommendations.

## Features

- **Natural language input**: Simply describe what anime you like or what you're looking for
- **Personalized recommendations**: Get 3-5 anime recommendations based on your preferences
- **Detailed information**: Each recommendation includes title, type, genre, and explanation
- **Vector search**: Utilizes embeddings to find similar anime titles
- **Hybrid approach**: Combines database information with LLM knowledge

## Project Structure
```
anime-recommender/
├── anime_rec.py          # Main Streamlit application
├── vector.py             # Vector database retriever setup
├── agents.py             # Script to create the vector database
├── animedata.csv         # Source data for anime information
├── chroma_langchain_db/  # Generated vector database (created by agents.py)
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

## Installation

1. Clone the repository:
   `git clone https://github.com/Anjila-26/AnimeRAG-recommendation-.git`
   `cd AnimeRAG-recommendation-`

2. Install dependencies
`pip install -r requirements.txt`

3. Install Ollama following the instructions at [ollama.ai](https://ollama.com/)

4. Pull the required models:
For me it's
`ollama pull llama3.2:1b`
`ollama pull mxbai-embed-large`
