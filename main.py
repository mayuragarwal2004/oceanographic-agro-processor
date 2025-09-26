"""
Main entry point for the Oceanographic Data Analysis AI System
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from agents.config import ConfigManager
from agents.orchestrator import AgentOrchestrator
from utils.query_logging import get_query_logger

async def main():
    """Main entry point for the system"""
    
    print("🌊 Oceanographic Data Analysis AI System")
    print("=" * 50)
    
    # Load configuration
    try:
        config = ConfigManager.load_from_env()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print("\nMake sure you have a .env file with the required variables:")
        print("GROQ_API_KEY=your_groq_api_key")
        print("DATABASE_HOST=localhost")
        print("DATABASE_PORT=5432")
        print("DATABASE_NAME=oceanographic_data")  
        print("DATABASE_USER=your_username")
        print("DATABASE_PASSWORD=your_password")
        return
    
    # Initialize orchestrator
    try:
        orchestrator = AgentOrchestrator(config)
        print("✅ Agent orchestrator initialized")
        
        # Perform health check
        health_status = await orchestrator.health_check()
        print(f"🏥 System health: {health_status['orchestrator']}")
        
        if health_status['orchestrator'] != 'healthy':
            print("⚠️  Some agents are not healthy:")
            for agent_name, status in health_status['agents'].items():
                if isinstance(status, dict) and status.get('status') != 'healthy':
                    print(f"   - {agent_name}: {status.get('error', 'unknown error')}")
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return
    
    # Interactive mode
    print("\n🔍 Enter your oceanographic queries (type 'quit' to exit):")
    print("Examples:")
    print("  - Average Temperature in the Indian Ocean")
    print("  - Compare salinity between Atlantic and Indian")  
    print("  - Show temperature trends in Indian Ocean from 2020 to 2023")
    print()
    
    session_id = f"interactive_{asyncio.get_event_loop().time()}"
    
    while True:
        try:
            # Get user query
            query = input("🌊 Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
            
            print(f"\n🔄 Processing: {query}")
            print("⏱️  This may take a few moments...\n")
            
            # Process query
            context = {
                'session_id': session_id,
                'original_query': query,
                'user_preferences': {
                    'style': 'general',
                    'detail_level': 'medium'
                }
            }
            
            result = await orchestrator.process(query, context)
            
            # Create a query logger for logging terminal output (same query should access same log file)
            query_logger_manager = get_query_logger()
            terminal_logger = query_logger_manager.get_query_logger(query, session_id)
            
            if result.success:
                success_msg = "✅ Analysis completed successfully!\n"
                print(success_msg)
                terminal_logger.info(success_msg.strip())
                
                # Display response
                response_data = result.data
                response_text = response_data.get('response', 'No response generated')
                results_header = "📋 Results:"
                separator = "-" * 40
                print(results_header)
                print(separator)
                print(response_text)
                
                terminal_logger.info(results_header)
                terminal_logger.info(separator)
                terminal_logger.info(response_text)
                
                # Show execution summary
                exec_summary = response_data.get('execution_summary', {})
                if exec_summary:
                    exec_time_msg = f"\n⏱️  Execution time: {exec_summary.get('total_time', 0):.1f}s"
                    agents_msg = f"🤖 Agents used: {', '.join(exec_summary.get('agents_executed', []))}"
                    print(exec_time_msg)
                    print(agents_msg)
                    
                    terminal_logger.info(exec_time_msg.strip())
                    terminal_logger.info(agents_msg)
                
                # Show available visualizations
                visualizations = response_data.get('visualizations', {})
                if visualizations:
                    viz_msg = f"\n📊 Visualizations created: {', '.join(visualizations.keys())}"
                    print(viz_msg)
                    terminal_logger.info(viz_msg.strip())
                
                # Show validation score
                validation = response_data.get('validation_report', {})
                if validation and 'overall_score' in validation:
                    score = validation['overall_score']
                    approval = validation.get('approved', False)
                    status = "✅ Approved" if approval else "⚠️  Has issues"
                    quality_msg = f"🔍 Data quality: {score:.1f}% {status}"
                    print(quality_msg)
                    terminal_logger.info(quality_msg)
                
            else:
                error_msg = "❌ Analysis failed:"
                print(error_msg)
                terminal_logger.error(error_msg)
                    
                for error in result.errors:
                    error_detail = f"   - {error}"
                    print(error_detail)
                    terminal_logger.error(error_detail)
            
            separator_line = "\n" + "=" * 50
            print(separator_line)
            terminal_logger.info("=" * 50)
            # Cleanup the logger now that we're done
            query_logger_manager.close_query_logger(terminal_logger)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue

def create_sample_env():
    """Create a sample .env file"""
    
    env_content = """# Oceanographic Data Analysis AI System Configuration

# GROQ API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=oceanographic_data
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password

# Optional: LLM Configuration
DEFAULT_LLM_PROVIDER=groq
DEFAULT_MODEL=llama-3.1-8b-instant
MAX_TOKENS=2048
TEMPERATURE=0.3

# Optional: System Configuration  
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=3
TIMEOUT_PER_AGENT=300
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(env_content)
        print(f"📝 Created sample .env file: {env_file.absolute()}")
        print("Please edit it with your actual configuration values.")
        return True
    return False

if __name__ == "__main__":
    # Check for .env file
    if not Path(".env").exists():
        print("⚙️  No .env file found")
        create_sample_env()
        print("\nPlease configure your .env file and run again.")
        sys.exit(1)
    
    # Run the main application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)