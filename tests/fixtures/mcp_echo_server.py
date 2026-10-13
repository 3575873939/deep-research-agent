"""最小 MCP server（stdio transport）：供测试用，不依赖任何外部服务。

暴露两个只读工具，用于验证「连接 → 列工具 → 调用 → 解析结果」的完整往返。
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo")


@mcp.tool()
def echo(text: str) -> str:
    """原样回显输入文本。"""
    return f"echo: {text}"


@mcp.tool()
def word_count(text: str) -> str:
    """统计输入文本的词数。"""
    return str(len(text.split()))


if __name__ == "__main__":
    mcp.run()
