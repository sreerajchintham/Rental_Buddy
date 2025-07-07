# Rental Buddy: AI-Powered Rental Property Search

**Rental Buddy** is an AI-driven real estate assistant designed to streamline the process of finding rental properties. Built as a Google Capstone Project, it leverages Retrieval-Augmented Generation (RAG) with Google's Generative AI, ChromaDB vector database, and LangChain/LangGraph for a conversational, user-friendly property search experience. This project processes a dataset of 10,000 rental listings, enabling semantic searches, property management, and visit planning.

## 📝 Overview

This project is implemented in a Jupyter notebook, combining data processing, vector embeddings, and a conversational AI workflow to assist users in finding and organizing rental properties. Key functionalities include:

- **Data Processing**: Loads and cleans a 10,000-row rental dataset, preparing it for semantic search.
- **Semantic Search**: Uses Google's `text-embedding-004` model to generate embeddings and ChromaDB for efficient querying.
- **Conversational Workflow**: Employs LangChain and LangGraph to manage multi-turn conversations, allowing users to search, save, remove, and group properties for visits.
- **Visit Planning**: Clusters properties geographically using KMeans and generates Google Maps URLs and summary cards for trip planning.

**Note**: This README covers the full notebook (33 cells), including data setup, embedding generation, vector storage, tool definitions, and the LangGraph conversational workflow.

## ✨ Key Features

- **Kaggle Dataset Integration**: Imports the `rental-10k-dataset` (10,000 rental listings) via `kagglehub`.
- **Robust Data Cleaning**: Handles missing values, standardizes columns, and drops irrelevant fields, reducing the dataset to 2,570 clean records with 19 features.
- **Semantic Search**: Embeds property listings using Google's `text-embedding-004` model and stores them in ChromaDB for natural language queries (e.g., "2 bedroom apartment in Chicago under $2000 with parking").
- **Structured Data Models**: Uses Pydantic (`PropertyJson`) for property validation and `TypedDict` (`Pstate`, `TripDict`) for state management.
- **Conversational AI**: Implements a stateful LangGraph workflow with tools for querying, saving, removing, and grouping properties, powered by `gemini-2.0-flash`.
- **Visit Planning**: Clusters properties into visit groups using KMeans and generates Google Maps URLs and detailed summary cards for each trip.
- **Error Handling**: Includes retry logic for API calls, validation for data integrity, and detailed logging for debugging.

## 🛠️ Technologies & Libraries

- **Python**: Core language (version 3.12.7).
- **kagglehub**: For dataset import from Kaggle.
- **pandas**, **numpy**: Data manipulation and numerical operations.
- **google-generativeai**: For embedding generation (`text-embedding-004`) and LLM interactions (`gemini-2.0-flash`).
- **chromadb**: Vector database for storing and querying property embeddings.
- **langchain**, **langchain-google-genai**, **langgraph**: Frameworks for conversational AI and stateful workflows.
- **pydantic**, **typing-extensions**: For data validation and type hints.
- **scikit-learn**: For KMeans clustering and `StandardScaler` in visit planning.
- **google-api-core**: Provides retry logic for robust API interactions.

## 🚀 Project Structure

The notebook is organized into the following sections:

1. **Kaggle Data Import** (Cells 1-2):
   - Uses `kagglehub` to authenticate and download the `rental-10k-dataset` (`apartments_for_rent_classified_10K.csv`).

2. **Package Installation** (Cell 3):
   - Installs required libraries: `chromadb`, `google-generativeai`, `langchain`, `langgraph`, `pydantic`, `scikit-learn`, etc.

3. **Library Imports & Constants** (Cell 4):
   - Imports libraries and defines constants (`GOOGLE_API_KEY`, `DATA_URL`, `COLLECTION_NAME`, `EMBEDDING_MODEL`).

4. **Data Loading & Cleaning** (Cells 5-6):
   - Loads the dataset (10,000 rows, 22 columns) with `pd.read_csv` (semicolon-separated, `cp1252` encoding).
   - Cleans data by stripping whitespace, dropping missing values (to 2,570 rows), and removing irrelevant columns (`category`, `has_photo`, `time`).

5. **ChromaDB Setup & Embedding** (Cells 7-8):
   - Initializes Google Generative AI client and a custom `GeminiEmbeddingFunction` with retry logic.
   - Embeds cleaned property listings (as JSON strings) and stores them in a ChromaDB collection (`Rental_Listings`) in batches of 100.

6. **ChromaDB Query Testing** (Cells 9-10):
   - Tests semantic search with a sample query ("I want 5 properties from Illinois"), retrieving 5 properties from Chicago, IL, with prices between $1,755 and $3,675.

7. **Data Models & State** (Cells 11-12):
   - Defines `PropertyJson` (Pydantic) for property validation, `TripDict` for visit groups, and `Pstate` for workflow state (messages, property set, trips).

8. **Tool Definitions** (Cells 13-18):
   - `query_properties`: Searches ChromaDB for properties matching user requirements.
   - `add_to_property_set`: Adds properties to the state’s `property_set`, ensuring no duplicates and validating with `PropertyJson`.
   - `remove_from_property_set`: Removes properties by index from `property_set`.
   - `generate_groups_for_visits`: Clusters properties into `n_trips` groups using KMeans based on latitude/longitude.
   - `generate_visit_plan`: Generates Google Maps URLs and summary cards for each trip using `gemini-2.0-flash`.

9. **LLM & System Instructions** (Cell 19):
   - Initializes `ChatGoogleGenerativeAI` with `gemini-2.0-flash` (temperature=0.7).
   - Defines `REAL_ESTATE_AGENT_SYSINT` with instructions for conversational flow and tool usage.

10. **Agent & Routing Logic** (Cells 20-21):
    - Defines `agent_node` to invoke the LLM and `should_continue` for routing between `agent`, `tools`, or `END`.
    - Constructs and compiles the LangGraph workflow with nodes (`agent`, `tools`) and conditional edges.

11. **Interactive Chat Session** (Cells 22-23):
    - Implements a loop for user interaction, processing inputs, invoking the graph, and displaying responses.
    - Supports commands like "quit" to exit and handles errors gracefully.

## 🎯 Key Results

- **Data Processing**: Successfully cleaned a 10,000-row dataset to 2,570 usable records with 19 relevant features.
- **Semantic Search**: Demonstrated accurate retrieval of properties (e.g., 5 listings from Illinois) using ChromaDB and embeddings.
- **Conversational Workflow**: Established a functional LangGraph pipeline with tools for searching, saving, removing, and grouping properties.
- **Visit Planning**: Enabled geographic clustering and generated actionable visit plans with map URLs and summaries (though map URL generation may rely on LLM outputs, which require validation).
- **Robustness**: Incorporated retry logic, validation, and extensive logging for reliable operation.

## ⚙️ How to Run

1. **Environment**:
   - Use Google Colab or Kaggle Notebook with GPU support (optional, as per notebook metadata: `T4` GPU, though not used in provided cells).

2. **Prerequisites**:
   - **Google API Key**: Set `GOOGLE_API_KEY` as a Kaggle secret (`kaggle_secrets.UserSecretsClient`) or environment variable.
   - **Dataset**: Ensure `rental-10k-dataset` is available in your Kaggle environment (`/kaggle/input/rental-10k-dataset/apartments_for_rent_classified_10K.csv`).

3. **Steps**:
   - Clone or download the notebook (`Copy_of_Google_CapStone_Project_Rental_Buddy.ipynb`).
   - Run cells sequentially to:
     - Install dependencies (Cell 3).
     - Load and clean data (Cells 5-6).
     - Set up ChromaDB and embed listings (Cells 7-8).
     - Define models, tools, and workflow (Cells 11-21).
     - Start the interactive chat (Cell 22).
   - For the chat session, enter natural language queries (e.g., "Find 2 bedroom apartments in Chicago under $2000") or commands like "quit".

4. **Example Interaction**:
   - Input: "Find me 2 bedroom apartments in Chicago between $1000 and $2000 with a gym."
   - Output: Lists matching properties, allows saving to `property_set`, removing by index, grouping into trips, and generating visit plans.
   - Note: The provided notebook output suggests issues with tool execution (pseudo-tool calls not executed) and potential data hallucination. Ensure proper tool integration in your environment.

## ⚠️ Known Issues & Limitations

- **Tool Execution**: The notebook output indicates that tool calls (e.g., `query_properties`) are output as code blocks rather than executed, suggesting a potential issue with `langchain` tool binding or LLM configuration.
- **Data Consistency**: Some responses include fabricated property details (e.g., "Luxury 2 bed 2 bath in Streeterville!") not present in the dataset, indicating possible LLM hallucination.
- **Validation Errors**: Adding properties to `property_set` may fail due to missing `id` fields in fabricated data, requiring stricter validation or real dataset integration.
- **Map URL Generation**: Relies on `gemini-2.0-flash` for Google Maps URLs, which may produce inconsistent or invalid results without proper validation.
- **Partial Notebook**: The provided README was based on Cells 1-14 initially, but this updated version accounts for all 33 cells. Ensure all cells are present for full functionality.

## 🔮 Future Improvements

- **Fix Tool Execution**: Debug and ensure proper tool integration with `langchain` and `langgraph` to execute `query_properties`, `add_to_property_set`, etc., correctly.
- **Prevent Hallucination**: Enforce stricter reliance on dataset metadata and tool outputs to avoid fabricated property details.
- **Enhance Map URLs**: Validate and refine Google Maps URL generation, possibly integrating Google Maps API for more reliable results.
- **User Interface**: Develop a front-end (e.g., using Streamlit or Flask) to make the assistant more accessible beyond the notebook.
- **Advanced Features**: Add filters for amenities, pet policies, or proximity to landmarks, and optimize clustering for travel time.

## 📚 Dataset Details

- **Source**: `chinthamsreeraj/rental-10k-dataset` on Kaggle.
- **File**: `apartments_for_rent_classified_10K.csv` (semicolon-separated, `cp1252` encoding).
- **Initial Shape**: 10,000 rows, 22 columns.
- **Cleaned Shape**: 2,570 rows, 19 columns (after dropping missing values and irrelevant columns: `category`, `has_photo`, `time`).
- **Key Columns**: `id`, `title`, `price`, `bedrooms`, `bathrooms`, `cityname`, `state`, `latitude`, `longitude`, `amenities`, `pets_allowed`, `sqft`.

## 📜 License

This project is for educational purposes as part of the Google Capstone Project. The dataset and code are subject to the licenses of their respective sources (Kaggle dataset, Google APIs, etc.).

## 🙌 Acknowledgments

- **Google Generative AI**: For providing embedding and LLM capabilities.
- **Kaggle**: For hosting the dataset and notebook environment.
- **LangChain/LangGraph & ChromaDB**: For enabling conversational AI and vector storage.

**Disclaimer**: This project is a prototype and may require additional debugging for production use, particularly for tool execution and data consistency.
