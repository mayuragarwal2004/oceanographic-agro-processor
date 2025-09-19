"""
Configuration management for the multi-agent system
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from .llm.base import LLMProvider

# Load environment variables from .env file
load_dotenv()

@dataclass
class LLMConfig:
    provider: LLMProvider
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    max_retries: int = 3

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    
    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class AgentSystemConfig:
    """Configuration for the entire agent system"""
    llm_config: LLMConfig
    database_config: DatabaseConfig
    debug: bool = False
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour default
    max_query_complexity: int = 100
    default_geohash_precision: int = 7

class ConfigManager:
    """Manages configuration loading from environment variables and files"""
    
    @staticmethod
    def load_from_env() -> AgentSystemConfig:
        """Load configuration from environment variables"""
        
        # LLM Configuration
        provider_str = os.getenv("LLM_PROVIDER", "groq").lower()
        provider = LLMProvider(provider_str)
        
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is required")
        
        model = os.getenv("LLM_MODEL")
        if not model:
            # Default models per provider
            model_defaults = {
                LLMProvider.GROQ: "llama3-8b-8192",
                LLMProvider.OPENAI: "gpt-4-turbo-preview",
                LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229",
                LLMProvider.GEMINI: "gemini-pro"
            }
            model = model_defaults.get(provider, "llama3-8b-8192")
        
        llm_config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4000")) if os.getenv("LLM_MAX_TOKENS") else None,
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3"))
        )
        
        # Database Configuration
        database_config = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "ocean"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "")
        )
        
        return AgentSystemConfig(
            llm_config=llm_config,
            database_config=database_config,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            enable_caching=os.getenv("ENABLE_CACHING", "true").lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            max_query_complexity=int(os.getenv("MAX_QUERY_COMPLEXITY", "100")),
            default_geohash_precision=int(os.getenv("GEOHASH_PRECISION", "7"))
        )
    
    @staticmethod
    def create_sample_env_file(filepath: str = ".env"):
        """Create a sample .env file with all configuration options"""
        
        sample_content = """# LLM Configuration
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL=mixtral-8x7b-32768
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000
LLM_TIMEOUT=30
LLM_MAX_RETRIES=3

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ocean
DB_USER=postgres
DB_PASSWORD=your_password_here

# System Configuration
DEBUG=false
ENABLE_CACHING=true
CACHE_TTL=3600
MAX_QUERY_COMPLEXITY=100
GEOHASH_PRECISION=7

# Optional: Alternative LLM Providers
# For OpenAI:
# LLM_PROVIDER=openai
# LLM_API_KEY=your_openai_api_key
# LLM_MODEL=gpt-4-turbo-preview

# For Anthropic:
# LLM_PROVIDER=anthropic
# LLM_API_KEY=your_anthropic_api_key
# LLM_MODEL=claude-3-sonnet-20240229

# For Gemini:
# LLM_PROVIDER=gemini
# LLM_API_KEY=your_gemini_api_key
# LLM_MODEL=gemini-pro
"""
        
        with open(filepath, 'w') as f:
            f.write(sample_content)
        
        print(f"Sample .env file created at {filepath}")
        print("Please edit it with your actual API keys and configuration.")

# Load configuration on import
try:
    config = ConfigManager.load_from_env()
except Exception as e:
    print(f"Warning: Could not load configuration from environment: {e}")
    print("Creating sample .env file...")
    ConfigManager.create_sample_env_file()
    config = None