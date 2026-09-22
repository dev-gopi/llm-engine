import pytest

from agents.mcp import MCPExecutionContract, MCPServerConfig


def test_mcp_contract_requires_transport():
    with pytest.raises(ValueError): MCPExecutionContract(MCPServerConfig(server_label="x"))

def test_mcp_authorization_requires_approval():
    c=MCPExecutionContract(MCPServerConfig(server_label="x",command="echo",allowed_tools=frozenset({"tool"}),approval_required=True))
    with pytest.raises(PermissionError): c.authorization.authorize("tool")
