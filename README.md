Here's a comprehensive `README.md` file for your Google Colab notebook, based on the provided content.

---

# Google Capstone Project: Rental Buddy

An AI-powered assistant for finding rental properties, leveraging Google's Generative AI, vector databases (ChromaDB), and LangChain/LangGraph for conversational search.

## 📝 Overview

This Google Colab notebook outlines the foundational steps for building "Rental Buddy," an intelligent assistant designed to help users find rental properties. The project focuses on integrating various AI technologies to create a robust and interactive property search experience.

The notebook covers:
*   Loading and comprehensive cleaning of a large rental property dataset.
*   Generating high-quality text embeddings for property listings using Google's `text-embedding-004` model.
*   Storing and managing these embeddings efficiently in a ChromaDB vector database for semantic search capabilities.
*   Defining structured data models (using Pydantic) to represent property information and the conversational state for a LangGraph-based AI workflow.
*   Demonstrating initial query functionality against the ChromaDB to retrieve semantically similar properties.

**Note**: This README is based on the provided notebook content up to Cell 14. The full LangGraph workflow, tool definitions, and further conversational AI logic are expected to be defined in subsequent, unprovided cells (the notebook has 33 cells in total).

## ✨ Key Features & Functionality

*   **Kaggle Data Integration**: Seamlessly imports datasets directly from Kaggle.
*   **Robust Data Preprocessing**: Cleans raw rental data by handling missing values, standardizing column names, and dropping irrelevant features.
*   **Google Generative AI Embeddings**: Utilizes Google's state-of-the-art `text-embedding-004` model to convert property descriptions into dense vector representations.
*   **ChromaDB Vector Store**: Employs ChromaDB as a local vector database for efficient storage and retrieval of embedded property listings, enabling semantic search.
*   **Custom Embedding Function**: Implements a custom `GeminiEmbeddingFunction` for ChromaDB, complete with retry mechanisms for API robustness.
*   **Structured Data Modeling**: Uses Pydantic for defining `PropertyJson` models, ensuring data validation and consistency for property details. `TypedDict` is used for managing the conversational workflow state (`Pstate`).
*   **Semantic Search Capability**: Demonstrates the ability to query the vector database using natural language, retrieving properties based on semantic similarity.
*   **Foundations for Conversational AI**: Sets up the basic data structures and imports necessary libraries (`langchain`, `langgraph`) for building a multi-turn, stateful conversational agent.

## 🛠️ Technologies & Libraries Used

*   **Python**: Core programming language.
*   **kagglehub**: For fetching Kaggle datasets.
*   **pandas**: For data manipulation and analysis.
*   **numpy**: For numerical operations.
*   **google-generativeai**: Google's SDK for interacting with Gemini models, specifically for embedding generation.
*   **chromadb**: An open-source vector database for storing and querying embeddings.
*   **langchain**: Framework for developing applications powered by large language models.
*   **langchain-google-genai**: LangChain integration with Google's Generative AI models.
*   **langgraph**: A library built on LangChain for creating stateful, multi-actor applications with LLMs using graphs.
*   **pydantic**: For data validation, settings management, and serialization using Python type hints.
*   **typing-extensions**: Provides backports of features from future Python versions and experimental types.
*   **scikit-learn**: (Imported, but not explicitly used in the provided cells) For potential future machine learning tasks like clustering (`KMeans`) and data scaling (`StandardScaler`).
*   **google-api-core**: Core utilities for Google APIs, including retry logic.

## 🚀 Main Sections & Steps

The notebook progresses through the following key stages:

1.  ### Kaggle Data Source Import
    (`Cell 1`, `Cell 2`)
    Initializes `kagglehub` and downloads the `rental-10k-dataset` required for the project.

2.  ### Initial Setup: Installing Required Packages
    (`Cell 3`, `Cell 4`)
    Installs all necessary Python libraries (`chromadb`, `google-generativeai`, `langchain`, `langgraph`, `pydantic`, `scikit-learn`, etc.) using `pip`.

3.  ### Importing Libraries and Defining Constants
    (`Cell 5`, `Cell 6`)
    Imports all required modules (e.g., `pandas`, `genai`, `chromadb`, `langchain_core`) and defines global constants such as `GOOGLE_API_KEY` (retrieved from Kaggle secrets), the dataset `DATA_URL`, `COLLECTION_NAME` for ChromaDB, and the `EMBEDDING_MODEL`.

4.  ### Loading and Cleaning the Rental Dataset
    (`Cell 7`, `Cell 8`)
    Loads the `apartments_for_rent_classified_10K.csv` file, performs initial data cleaning by stripping whitespace from column names, dropping rows with missing values, and removing irrelevant columns. The dataset is prepared for further processing.

5.  ### Setting Up ChromaDB and Embedding Property Listings
    (`Cell 9`, `Cell 10`)
    Initializes the Google Generative AI client and defines `GeminiEmbeddingFunction`—a custom embedding function for ChromaDB that uses `text-embedding-004`. The cleaned property listings are then batched, embedded, and added to the `Rental_Listings` collection in ChromaDB.

6.  ### Testing ChromaDB Query Functionality
    (`Cell 11`, `Cell 12`)
    Performs a sample query (e.g., "I want 5 properties from Illinois") against the `Rental_Listings` collection to verify that the embeddings and ChromaDB setup are working correctly, demonstrating semantic retrieval.

7.  ### Defining Data Models and Graph State
    (`Cell 13`, `Cell 14`)
    Defines Pydantic models (`PropertyJson`) for structured representation of property details and `TypedDict` (`Pstate`) to manage the state of the LangGraph workflow, including conversation history and selected properties.

## 🎯 Key Insights & Results

*   **Successful Data Ingestion**: A robust pipeline for loading, cleaning, and preparing a large dataset of rental properties.
*   **Efficient Semantic Search**: Established a functional ChromaDB vector database capable of performing semantic searches on property descriptions using Google's powerful embedding models.
*   **Foundational Architecture**: Built the essential data models and integrated key libraries (`langchain`, `langgraph`, `google-generativeai`, `chromadb`) that form the backbone of a sophisticated conversational AI application.
*   **Verified Retrieval**: Confirmed that the vector database can accurately retrieve relevant property listings based on natural language queries, showcasing the power of embeddings for search.

## ⚙️ How to Use/Run the Notebook

To run this notebook and recreate the analysis and setup:

1.  **Environment**: It is recommended to run this notebook in a Google Colab environment or a Kaggle Notebook environment.
2.  **Google API Key**:
    *   Ensure you have a `GOOGLE_API_KEY` set up as a secret in your Kaggle environment (named `GOOGLE_API_KEY`). The notebook uses `kaggle_secrets.UserSecretsClient` to retrieve this key.
    *   If running outside Kaggle, you will need to manually set the `GOOGLE_API_KEY` environment variable or hardcode it (though not recommended for production).
3.  **Dataset**: The notebook expects the `rental-10k-dataset` (specifically `apartments_for_rent_classified_10K.csv`) to be available as an input dataset in your Kaggle environment.
4.  **Execute Cells**: Run all cells sequentially from top to bottom. The first few cells will handle package installations and data source imports.

**Disclaimer**: As noted, this README is based on a partial notebook content (Cells 1-14 out of 33). The full application, including the complete LangGraph definition, tool usage, and final conversational flow, would be implemented in subsequent cells not provided in this excerpt.