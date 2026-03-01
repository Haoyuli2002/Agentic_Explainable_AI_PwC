import sys
import os

# Add the project root to sys.path so 'agent' can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import app

if __name__ == "__main__":
    png_data = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)