from typing import Dict, Any, Optional
from app.mcp.github_client import GitHubMCPClient

class MCPGateway:
    """
    Model Context Protocol (MCP) Gateway:
    Provides standardized tool access to external resources (GitHub MCP, Postgres MCP).
    """
    
    @staticmethod
    def inspect_candidate_code(github_url: Optional[str]) -> Dict[str, Any]:
        return GitHubMCPClient.analyze_repository(github_url)
