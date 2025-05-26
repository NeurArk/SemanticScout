# PR8: Visualization & Analytics - Detailed Implementation Guide

## Overview
This PR implements document visualization features including similarity plots and relationship networks.

## Prerequisites
- PR2-7 completed
- plotly, networkx, umap-learn installed

## Key Components

### 1. UMAP Visualization (`core/visualizer.py`)

```python
import numpy as np
from umap import UMAP
import plotly.graph_objects as go
from typing import List, Dict, Any
from core.models.document import Document, DocumentChunk

class DocumentVisualizer:
    """Create document visualizations."""
    
    def create_similarity_plot(self, documents: List[Dict], embeddings: List[List[float]]) -> go.Figure:
        """Create 2D scatter plot of document similarity."""
        
        # Reduce dimensions with UMAP
        reducer = UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(np.array(embeddings))
        
        # Create plotly figure
        fig = go.Figure(data=[
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers+text',
                marker=dict(size=10, color='blue'),
                text=[d['filename'] for d in documents],
                textposition="top center"
            )
        ])
        
        fig.update_layout(
            title="Document Similarity Map",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            hovermode='closest'
        )
        
        return fig
```

### 2. Network Graph

```python
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def create_document_network(embeddings: List[List[float]], 
                          documents: List[Dict], 
                          threshold: float = 0.8) -> go.Figure:
    """Create network graph of document relationships."""
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, doc in enumerate(documents):
        G.add_node(i, label=doc['filename'])
    
    # Add edges based on similarity
    for i in range(len(documents)):
        for j in range(i+1, len(documents)):
            if similarity_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=similarity_matrix[i, j])
    
    # Convert to plotly
    pos = nx.spring_layout(G)
    
    # Create edges
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='gray'),
            hoverinfo='none'
        ))
    
    # Create nodes
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[G.nodes[node]['label'] for node in G.nodes()],
        textposition="top center",
        marker=dict(size=20, color='lightblue', line=dict(width=2))
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        title="Document Relationship Network",
        showlegend=False,
        hovermode='closest'
    )
    
    return fig
```

## Integration with Gradio

```python
# In app.py visualization tab
with gr.TabItem("📊 Visualize"):
    plot_output = gr.Plot(label="Document Visualization")
    
    with gr.Row():
        viz_type = gr.Radio(
            choices=["Similarity Map", "Network Graph", "Cluster Analysis"],
            value="Similarity Map",
            label="Visualization Type"
        )
        
        refresh_viz_btn = gr.Button("🔄 Generate Visualization")
    
    refresh_viz_btn.click(
        fn=generate_visualization,
        inputs=[viz_type],
        outputs=[plot_output]
    )
```

## Success Criteria

1. ✅ UMAP visualization shows document relationships
2. ✅ Network graph displays similarity connections
3. ✅ Interactive plots with zoom/pan
4. ✅ Handles 100+ documents smoothly
5. ✅ Integrates seamlessly with Gradio UI