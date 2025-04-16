# Anime Recommendation System

A personalized anime recommendation system powered by LangChain, Ollama, and vector search.

![Anime Recommender](https://github.com/username/anime-recommender/blob/main/screenshots/app_screenshot.png)

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

## Database Setup
**Important:** The vector database only needs to be created once. After initial setup, the application will use the existing database.
To set up the database:

1. Make sure your anime data is in a CSV file named animedata.csv with at least these columns:

name - Anime title
Type - Format (TV, Movie, OVA, etc.)
Plot Summary - Brief description
Genre - Categories/genres
Status - Airing status
Other name - Alternative titles (optional)

2. Run the database creation script:
`python agents.py`

This will:

Create embeddings for each anime entry
Store them in a Chroma vector database
Display progress information
Save the database to ./chroma_langchain_db/

The process may take several minutes depending on the size of your dataset and your hardware.
Running the Application
Once the database is created, you can run the Streamlit application:
`streamlit run anime_rec.py`

The application will open in your default web browser.
How It Works

**User Input**: The user enters their anime preferences or describes what they're looking for
**Vector Search**: The system uses the query to find the most similar anime entries in the vector database
**LLM Processing**: The retrieved anime information is sent to the LLM along with the user query
**Recommendation Generation**: The LLM generates personalized recommendations based on both:

The specific anime information from the database
Its general knowledge about anime relationships and similarities


Result Display: The recommendations are presented to the user in a clean, formatted interface

## Important Notes

Database Creation: The vector database is only created once. After the initial setup, the application simply reads from the existing database.
Hybrid Approach: The system combines factual information from your database with the LLM's understanding of anime relationships.
Performance: Retrieval and recommendation generation may take a few seconds depending on your hardware.
Model Selection: The application uses Llama 3.2 (1B parameter version) for recommendations and mxbai-embed-large for embeddings.


