# Oceanographic Data Analysis AI System

A sophisticated multi-agent AI system for analyzing oceanographic data using natural language queries. Built specifically for NOAA GADR (Global Archive of Argo Data Repository) dataset analysis.

## 🌊 Features

### Multi-Agent Architecture
- **Query Understanding Agent**: Natural language processing and intent extraction
- **Geospatial Agent**: Location resolution and spatial query handling  
- **Data Retrieval Agent**: Optimized SQL generation and database querying
- **Analysis Agent**: Statistical analysis, trend detection, and anomaly identification
- **Visualization Agent**: Interactive maps, charts, and plots
- **Critic Agent**: Scientific validation and quality assurance
- **Conversation Agent**: Natural language response generation
- **Orchestrator**: Workflow coordination and execution management

### Capabilities
- 🗣️ **Natural Language Queries**: Ask questions in plain English
- 🗺️ **Spatial Analysis**: Location-based oceanographic analysis
- 📊 **Statistical Analysis**: Trends, correlations, and anomaly detection
- 📈 **Interactive Visualizations**: Maps, time series, depth profiles
- 🔍 **Data Quality Validation**: Scientific accuracy checks
- 💬 **Conversational Interface**: Follow-up questions and clarifications

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database with NOAA GADR data
- GROQ API key (or other supported LLM provider)

### Installation

1. **Clone and setup**:
```bash
cd /home/mayur/mayur/SIH25/Agro/040float
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
cp .env.sample .env  # Edit with your settings
```

Required environment variables:
```env
GROQ_API_KEY=your_groq_api_key
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=oceanographic_data
DATABASE_USER=your_username  
DATABASE_PASSWORD=your_password
```

3. **Setup database** (using the provided schema.prisma):
```bash
# Setup your PostgreSQL database with the Prisma schema
# The schema is optimized for NOAA GADR data structure
```

4. **Run the system**:
```bash
python main.py
```

## 💬 Example Queries

The system handles various types of oceanographic queries:

### Basic Data Queries
```
Temperature in the Pacific Ocean last month
Show salinity data for the Atlantic Ocean in 2023
What's the pressure at 1000m depth in the Indian Ocean?
```

### Comparative Analysis
```
Compare temperature between Pacific and Atlantic oceans
How does salinity vary between different ocean basins?
Temperature differences between 2020 and 2023
```

### Trend Analysis
```
Temperature trends in the Pacific Ocean from 2020 to 2023
Has salinity been increasing in the Atlantic?
Show seasonal patterns in ocean temperature
```

### Statistical Analysis
```
Average temperature by depth in the Pacific
Find temperature anomalies in 2023
Correlation between temperature and salinity
```

## 🏗️ Architecture

### System Components

```
User Query
    ↓
┌─────────────────┐
│   Orchestrator  │ ←→ Coordinates all agents
└─────────────────┘
    ↓
┌─────────────────┐
│ Query Understanding │ ←→ Extract entities & intent  
└─────────────────┘
    ↓
┌─────────────────┐
│   Geospatial    │ ←→ Resolve locations
└─────────────────┘
    ↓
┌─────────────────┐
│ Data Retrieval  │ ←→ Generate SQL & fetch data
└─────────────────┘
    ↓
┌─────────────────┐
│    Analysis     │ ←→ Statistical computations
└─────────────────┘
    ↓
┌─────────────────┐
│ Visualization   │ ←→ Create charts & maps
└─────────────────┘
    ↓
┌─────────────────┐
│     Critic      │ ←→ Validate results
└─────────────────┘
    ↓
┌─────────────────┐
│ Conversation    │ ←→ Generate response
└─────────────────┘
    ↓
Natural Language Response + Visualizations
```

### Database Schema

The system uses an optimized PostgreSQL schema for NOAA GADR data:

- **Platform**: Argo float information
- **Profile**: Individual measurement profiles  
- **Measurement**: Temperature, salinity, pressure measurements

Key optimizations:
- Only includes parameters actually present in NOAA data (TEMP, PSAL, PRES)
- Spatial and temporal indexing for fast queries
- Quality control flag support

## 🔧 Configuration

### Environment Variables

```env
# LLM Configuration
GROQ_API_KEY=your_api_key
DEFAULT_LLM_PROVIDER=groq
DEFAULT_MODEL=llama-3.1-8b-instant

# Database Configuration  
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=oceanographic_data
DATABASE_USER=username
DATABASE_PASSWORD=password

# System Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=3
TIMEOUT_PER_AGENT=300
```

### Supported LLM Providers

- **GROQ** (default): Fast inference with Llama models
- **OpenAI**: GPT models (requires OpenAI API key)
- **Anthropic**: Claude models (requires Anthropic API key)  
- **Google**: Gemini models (requires Google API key)

## 📊 Data Requirements

### NOAA GADR Dataset

The system is designed for NOAA's Global Archive of Argo Data Repository:

- **Format**: NetCDF files (.nc)
- **Structure**: Profile-based measurements
- **Parameters**: Temperature (TEMP), Salinity (PSAL), Pressure (PRES)
- **Coverage**: Global ocean, 2000-present

### Data Ingestion

Use the provided `injectdata.py` script to populate your database:

```bash
python injectdata.py --data-dir ./argo_data --batch-size 100
```

## 🧪 Testing

### Example Test Queries

```python
# Test basic functionality
query = "Temperature in Pacific Ocean"
result = await orchestrator.process(query)

# Test complex analysis
query = "Compare Atlantic and Pacific temperature trends 2020-2023"
result = await orchestrator.process(query)
```

### Health Checks

```python
# Check system health
health = await orchestrator.health_check()
print(health['orchestrator'])  # healthy/degraded/unhealthy
```

## 🚨 Known Limitations

1. **Data Dependencies**: Requires properly formatted NOAA GADR data
2. **LLM Dependencies**: Requires active API key for LLM provider
3. **Database Performance**: Large queries may take time on extensive datasets
4. **Missing Dependencies**: Some visualization libraries need manual installation

### Missing Python Packages

The following packages may need separate installation:
```bash
pip install asyncpg scipy scikit-learn plotly folium
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Install missing dependencies
2. **Database Connection**: Check PostgreSQL connection and credentials
3. **LLM API Errors**: Verify API key and provider settings
4. **Query Timeouts**: Increase `TIMEOUT_PER_AGENT` for complex queries

### Debugging

Enable detailed logging:
```env
LOG_LEVEL=DEBUG
```

## 🛠️ Development

### Adding New Agents

1. Inherit from `BaseAgent`
2. Implement `process()` method
3. Add to orchestrator workflow
4. Update dependencies

### Extending LLM Support

1. Inherit from `BaseLLMConnector`
2. Implement provider-specific methods
3. Register with `LLMConnectorFactory`

## 📚 API Reference

### Core Classes

- `AgentOrchestrator`: Main system coordinator
- `BaseAgent`: Base class for all agents
- `AgentSystemConfig`: Configuration management
- `LLMConnectorFactory`: LLM provider abstraction

### Agent Methods

All agents implement:
```python
async def process(self, input_data: Any, context: Dict[str, Any]) -> AgentResult
def get_capabilities(self) -> Dict[str, Any]
async def health_check(self) -> Dict[str, Any]
```

## 🤝 Contributing

1. Follow the established agent architecture
2. Add comprehensive error handling
3. Include scientific validation
4. Update documentation

## 📄 License

This project is part of the SIH25 (Smart India Hackathon 2025) submission for oceanographic data analysis.

## 🙏 Acknowledgments

- NOAA for the GADR dataset
- Argo program for oceanographic measurements
- GROQ for fast LLM inference
- Open source oceanographic community

---

Built with ❤️ for ocean data science and climate research.

## FloatChat: Natural Language Query System

FloatChat is a natural language interface for querying ocean data. It allows users to ask questions about ocean measurements in plain English and gets the relevant data from the MySQL database.

### Features

- Natural language understanding for ocean data queries
- Multiple implementation options:
  - Pattern-based: Using regex for simple queries (floatchat.py)
  - LLM-based: Using Mistral LLM via Ollama for more flexible understanding (floatchat_llm_fixed.py)
  - Gemini-based: Using Google's Gemini API for enhanced understanding (floatchat_gemini.py)
- Geocoding of location names to coordinates
- SQL query generation based on parsed natural language queries
- Automated dependency management and environment setup

### Installation for FloatChat

1. Ensure you have the required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Choose your implementation:
   - For the pattern-based implementation: No additional setup needed
   - For the LLM-based implementation: Install Ollama from [ollama.ai](https://ollama.ai)
   - For the Gemini-based implementation: Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Using the Gemini API Implementation

The Gemini API implementation provides the most advanced natural language understanding capabilities with minimal setup:

1. Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set your API key as an environment variable:
   ```
   # On Windows
   setx GOOGLE_API_KEY "your_api_key_here"
   
   # On Linux/macOS
   export GOOGLE_API_KEY="your_api_key_here"
   ```
3. Run the launcher script:
   ```
   python floatchat_gemini_launcher.py
   ```

The launcher will check for the required dependencies and API key, then start the FloatChat application.

### Example Usage

#### Pattern-based implementation

```python
from floatchat import generate_sql_from_question

# Example query
question = "Show me temperature data from the Indian Ocean at 100m depth"
sql = generate_sql_from_question(question)
print(sql)
```

#### Gemini-based implementation

```python
from floatchat_gemini import generate_sql_from_question

# Example query
question = "What's the salinity in the Bay of Bengal between January and December 2000?"
sql = generate_sql_from_question(question)
print(sql)
```

### Testing

Run the test script to verify the Gemini implementation:

```
python test_floatchat_gemini.py
```

### Implementation Details

#### Pattern-based Parser (floatchat.py)

Uses regular expressions to extract location, measurement type, depth, and time range from natural language queries.

#### Gemini-based Parser (floatchat_gemini.py)

Uses Google's Gemini API to understand natural language queries and extract structured information. Features include:

- Robust error handling
- Fallback to regex parser if Gemini API fails
- Multiple JSON parsing strategies
- Geocoding for location names
- Free tier access with Google AI Studio API key

## Dependencies

- Python 3.8+
- PyMySQL
- Geopy (for geocoding)
- Requests (for API communication)
- NetCDF4 (for data injection)
- google-generativeai (for Gemini API implementation)
