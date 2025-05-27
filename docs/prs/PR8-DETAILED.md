# PR8: Basic Analytics

## Overview

Add simple statistics display.

## Goal

Show basic system metrics that impress during demos without complexity.

## Simple Implementation

### Add to `app.py`

```python
def get_system_stats() -> str:
    """Get simple system statistics."""
    try:
        stats = vector_store.get_statistics()

        return f"""
        📊 **System Statistics**

        • Documents: {stats.get('total_documents', 0)}
        • Total Chunks: {stats.get('total_chunks', 0)}
        • Vector Store Size: {stats.get('collection_size', 0)}

        **Document Types:**
        • PDF: {stats.get('pdf_count', 0)}
        • DOCX: {stats.get('docx_count', 0)}
        • TXT: {stats.get('txt_count', 0)}
        """
    except:
        return "📊 Statistics unavailable"

# Add to Gradio interface
with gr.Tab("Analytics"):
    stats_display = gr.Markdown(get_system_stats())
    refresh_stats = gr.Button("Refresh Stats")

    refresh_stats.click(
        fn=get_system_stats,
        outputs=[stats_display]
    )
```

## What We're NOT Building

- ❌ Complex visualizations (UMAP, t-SNE)
- ❌ Interactive graphs
- ❌ Performance monitoring
- ❌ Usage analytics
- ❌ Export functionality

## If You Have Extra Time

Add a simple bar chart using Gradio's built-in plotting:

```python
import pandas as pd

def create_simple_chart():
    """Create simple document type chart."""
    data = {
        'Type': ['PDF', 'DOCX', 'TXT'],
        'Count': [5, 3, 2]
    }
    df = pd.DataFrame(data)

    return gr.BarPlot.update(
        value=df,
        x="Type",
        y="Count",
        title="Documents by Type"
    )
```

## Success Criteria

- [ ] Stats display without errors
- [ ] Numbers are accurate
- [ ] Doesn't slow down main interface
- [ ] Looks professional
