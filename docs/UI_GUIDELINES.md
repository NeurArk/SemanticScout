# SemanticScout - UI/UX Guidelines

**Version**: 1.0  
**Date**: May 2025  
**Status**: Design Ready

## 🎨 Design Philosophy

SemanticScout's interface embodies modern AI application design principles: **simplicity**, **intelligence**, and **professional elegance**. The UI prioritizes user productivity while showcasing advanced AI capabilities in an accessible, intuitive manner.

### Core Design Principles

1. **Clarity Over Complexity**: Clean, uncluttered interface with obvious next steps
2. **Intelligence Made Visible**: AI operations are transparent with clear feedback
3. **Professional Aesthetics**: Corporate-grade visual design suitable for client demos
4. **Responsive Interactions**: Immediate feedback for all user actions
5. **Accessibility First**: Usable by all skill levels without training

## 🎯 User Experience Strategy

### Primary User Journey
```
1. Document Upload → 2. Processing Feedback → 3. Search Interface → 4. Results Exploration → 5. Insights Discovery
```

### Interaction Patterns
- **Progressive Disclosure**: Advanced features revealed as needed
- **Contextual Guidance**: Just-in-time help and tooltips
- **Visual Feedback**: Clear status indicators for all operations
- **Error Prevention**: Validation and guidance before errors occur

## 🖼️ Visual Design System

### Color Palette

#### Primary Colors
- **Tech Blue**: `#1E3A8A` - Primary actions, headers, highlights
- **Deep Blue**: `#1E40AF` - Secondary actions, hover states
- **Light Blue**: `#EFF6FF` - Background highlights, success states

#### Secondary Colors
- **Sophisticated Gray**: `#6B7280` - Body text, secondary elements
- **Light Gray**: `#F9FAFB` - Background areas, card backgrounds
- **Dark Gray**: `#374151` - Headers, emphasis text

#### Accent Colors
- **Success Green**: `#10B981` - Success states, positive feedback
- **Warning Orange**: `#F59E0B` - Warnings, processing states
- **Error Red**: `#EF4444` - Errors, critical alerts
- **Info Blue**: `#3B82F6` - Information, neutral alerts

#### Usage Guidelines
```css
/* Primary Actions */
.primary-button { background: #1E3A8A; color: white; }
.primary-text { color: #1E3A8A; }

/* Status Indicators */
.success { color: #10B981; }
.warning { color: #F59E0B; }
.error { color: #EF4444; }
.info { color: #3B82F6; }

/* Text Hierarchy */
.heading { color: #111827; font-weight: 600; }
.body-text { color: #374151; }
.secondary-text { color: #6B7280; }
```

### Typography

#### Font Stack
- **Primary**: Inter (Google Fonts)
- **Code/Monospace**: Fira Code
- **Fallback**: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto

#### Type Scale
```css
/* Headings */
.text-4xl { font-size: 2.25rem; line-height: 2.5rem; } /* Main title */
.text-3xl { font-size: 1.875rem; line-height: 2.25rem; } /* Section headers */
.text-2xl { font-size: 1.5rem; line-height: 2rem; } /* Subsection headers */
.text-xl { font-size: 1.25rem; line-height: 1.75rem; } /* Card titles */

/* Body Text */
.text-lg { font-size: 1.125rem; line-height: 1.75rem; } /* Emphasized body */
.text-base { font-size: 1rem; line-height: 1.5rem; } /* Default body */
.text-sm { font-size: 0.875rem; line-height: 1.25rem; } /* Small text */
.text-xs { font-size: 0.75rem; line-height: 1rem; } /* Labels, captions */
```

### Spacing System

#### Spacing Scale (Tailwind-inspired)
```css
/* Spacing values */
.space-1 { margin/padding: 0.25rem; } /* 4px */
.space-2 { margin/padding: 0.5rem; }  /* 8px */
.space-3 { margin/padding: 0.75rem; } /* 12px */
.space-4 { margin/padding: 1rem; }    /* 16px */
.space-6 { margin/padding: 1.5rem; }  /* 24px */
.space-8 { margin/padding: 2rem; }    /* 32px */
.space-12 { margin/padding: 3rem; }   /* 48px */
```

## 🧩 Component Design Specifications

### Gradio Theme Configuration

```python
import gradio as gr

# Custom theme based on gr.themes.Soft()
semantic_scout_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
    font_mono=["Fira Code", "Consolas", "monospace"]
).set(
    # Color customizations
    button_primary_background_fill="#1E3A8A",
    button_primary_background_fill_hover="#1E40AF",
    button_primary_text_color="white",
    
    # Background colors
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F9FAFB",
    
    # Border and shadows
    border_color_primary="#E5E7EB",
    shadow_drop="0 1px 3px 0 rgba(0, 0, 0, 0.1)",
    
    # Text colors
    body_text_color="#374151",
    body_text_color_subdued="#6B7280",
)
```

### Layout Components

#### Main Application Layout
```python
with gr.Blocks(theme=semantic_scout_theme, title="SemanticScout") as app:
    # Header
    gr.Markdown("# 🔍 SemanticScout", elem_classes=["main-title"])
    gr.Markdown("*Intelligent Document Search & Analysis*", elem_classes=["subtitle"])
    
    # Main content tabs
    with gr.Tabs():
        with gr.Tab("📄 Documents"):
            # Document management interface
        with gr.Tab("🔍 Search"):
            # Search interface
        with gr.Tab("📊 Visualize"):
            # Visualization interface
        with gr.Tab("⚙️ Settings"):
            # Configuration interface
```

#### Document Upload Interface
```python
with gr.Row():
    with gr.Column(scale=2):
        file_upload = gr.File(
            label="Upload Documents",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt", ".md"],
            elem_classes=["upload-area"]
        )
        
    with gr.Column(scale=1):
        upload_status = gr.Textbox(
            label="Status",
            interactive=False,
            elem_classes=["status-display"]
        )
        
        upload_progress = gr.Progress()
```

#### Search Interface
```python
with gr.Row():
    with gr.Column(scale=4):
        search_box = gr.Textbox(
            label="Search Query",
            placeholder="Ask anything about your documents...",
            elem_classes=["search-input"]
        )
        
    with gr.Column(scale=1):
        search_button = gr.Button(
            "🔍 Search",
            variant="primary",
            elem_classes=["search-button"]
        )

# Advanced options (collapsible)
with gr.Accordion("Advanced Options", open=False):
    with gr.Row():
        result_limit = gr.Slider(
            minimum=5, maximum=50, value=10,
            label="Max Results"
        )
        similarity_threshold = gr.Slider(
            minimum=0.0, maximum=1.0, value=0.7,
            label="Similarity Threshold"
        )
```

#### Results Display
```python
with gr.Row():
    with gr.Column(scale=3):
        results_display = gr.DataFrame(
            headers=["Document", "Score", "Preview"],
            datatype=["str", "number", "str"],
            elem_classes=["results-table"]
        )
        
    with gr.Column(scale=2):
        document_preview = gr.Textbox(
            label="Document Preview",
            lines=10,
            interactive=False,
            elem_classes=["document-preview"]
        )
```

### Interactive Elements

#### Button Styles
```python
# Primary actions
primary_button = gr.Button(
    variant="primary",
    size="lg",
    elem_classes=["btn-primary"]
)

# Secondary actions  
secondary_button = gr.Button(
    variant="secondary",
    size="md",
    elem_classes=["btn-secondary"]
)

# Danger actions
danger_button = gr.Button(
    variant="stop",
    size="sm",
    elem_classes=["btn-danger"]
)
```

#### Input Components
```python
# Text inputs with professional styling
text_input = gr.Textbox(
    label="Label",
    placeholder="Helpful placeholder text...",
    elem_classes=["input-professional"]
)

# File uploads with drag-and-drop
file_input = gr.File(
    label="Upload Files",
    elem_classes=["upload-dropzone"]
)

# Sliders with clear labels
slider_input = gr.Slider(
    minimum=0, maximum=100, value=50,
    label="Parameter Name",
    info="Helpful description of what this controls",
    elem_classes=["slider-professional"]
)
```

## 📱 Responsive Design

### Breakpoint Strategy
```css
/* Mobile First Approach */
.container {
    /* Mobile: < 768px */
    padding: 1rem;
    grid-template-columns: 1fr;
}

@media (min-width: 768px) {
    /* Tablet: 768px - 1024px */
    .container {
        padding: 1.5rem;
        grid-template-columns: 1fr 1fr;
    }
}

@media (min-width: 1024px) {
    /* Desktop: > 1024px */
    .container {
        padding: 2rem;
        grid-template-columns: 2fr 1fr;
        max-width: 1200px;
        margin: 0 auto;
    }
}
```

### Mobile Adaptations
- **Touch-friendly targets**: Minimum 44px touch targets
- **Simplified navigation**: Collapsible menus and tabs
- **Optimized content**: Priority content first, secondary features hidden
- **Performance**: Reduced visual effects, optimized images

## 🎭 Animation and Interactions

### Micro-Interactions
```python
# Loading states with progress indicators
with gr.Row():
    status_indicator = gr.HTML(
        '<div class="loading-spinner"></div>',
        visible=False
    )
    status_text = gr.Textbox(
        value="Ready",
        interactive=False,
        elem_classes=["status-text"]
    )
```

### Transition Guidelines
- **Duration**: 200-300ms for most interactions
- **Easing**: `cubic-bezier(0.4, 0, 0.2, 1)` for natural feel
- **Performance**: CSS transforms over layout changes
- **Accessibility**: Respect `prefers-reduced-motion`

### CSS Animations
```css
/* Loading spinner */
.loading-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid #f3f3f3;
    border-top: 2px solid #1E3A8A;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Smooth transitions */
.transition-smooth {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Hover effects */
.btn-hover:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
```

## 📊 Data Visualization Guidelines

### Chart Styling
```python
# Plotly theme configuration
plotly_theme = {
    'layout': {
        'colorway': ['#1E3A8A', '#10B981', '#F59E0B', '#EF4444'],
        'font': {'family': 'Inter, sans-serif', 'size': 12},
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60}
    }
}

# Document similarity scatter plot
fig = go.Figure()
fig.update_layout(
    title="Document Similarity Landscape",
    xaxis_title="UMAP Dimension 1",
    yaxis_title="UMAP Dimension 2",
    **plotly_theme['layout']
)
```

### Network Visualization
```python
# NetworkX with Plotly styling
def create_similarity_network():
    fig = go.Figure()
    
    # Node styling
    node_trace = go.Scatter(
        mode='markers+text',
        marker=dict(
            size=12,
            color='#1E3A8A',
            line=dict(width=1, color='white')
        ),
        textfont=dict(size=10, color='#374151')
    )
    
    # Edge styling  
    edge_trace = go.Scatter(
        mode='lines',
        line=dict(width=1, color='#E5E7EB'),
        hoverinfo='none'
    )
    
    return fig
```

## 🚨 Error Handling & Feedback

### Error Message Design
```python
def show_error_message(message: str, error_type: str = "error"):
    """Display user-friendly error messages"""
    
    error_styles = {
        "error": {"color": "#EF4444", "icon": "❌"},
        "warning": {"color": "#F59E0B", "icon": "⚠️"},
        "info": {"color": "#3B82F6", "icon": "ℹ️"},
        "success": {"color": "#10B981", "icon": "✅"}
    }
    
    style = error_styles.get(error_type, error_styles["error"])
    
    return gr.Markdown(
        f"{style['icon']} **{message}**",
        elem_classes=[f"alert-{error_type}"]
    )
```

### Loading States
```python
def show_loading_state(operation: str):
    """Show loading feedback during operations"""
    
    return gr.HTML(f"""
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <span class="loading-text">Processing {operation}...</span>
        </div>
    """)
```

### Success Feedback
```python
def show_success_message(message: str, details: str = ""):
    """Display success notifications"""
    
    return gr.Markdown(f"""
        ✅ **{message}**
        {details if details else ""}
    """, elem_classes=["alert-success"])
```

## ♿ Accessibility Standards

### WCAG 2.1 Compliance
- **Color Contrast**: Minimum 4.5:1 for normal text, 3:1 for large text
- **Keyboard Navigation**: All functions accessible via keyboard
- **Screen Reader Support**: Proper ARIA labels and descriptions
- **Focus Management**: Visible focus indicators and logical tab order

### Implementation Guidelines
```python
# Accessible form components
accessible_input = gr.Textbox(
    label="Search Documents",
    info="Enter keywords or natural language questions",
    elem_id="search-input",
    elem_classes=["accessible-input"]
)

# Proper labeling
file_upload = gr.File(
    label="Upload Documents", 
    file_types=[".pdf", ".docx", ".txt"],
    elem_id="file-upload",
    accessible=True
)
```

### Accessibility CSS
```css
/* Focus indicators */
.accessible-input:focus {
    outline: 2px solid #1E3A8A;
    outline-offset: 2px;
}

/* Screen reader only content */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
    .btn-primary {
        border: 2px solid currentColor;
    }
}
```

## 📏 Design Specifications Summary

### Component Spacing
- **Micro spacing**: 4px, 8px (within components)
- **Macro spacing**: 16px, 24px, 32px (between components)
- **Section spacing**: 48px, 64px (between major sections)

### Interactive States
- **Default**: Base styling
- **Hover**: Subtle elevation or color change
- **Active**: Pressed state indication
- **Focus**: Clear focus ring
- **Disabled**: Reduced opacity (0.6) with cursor indication

### Content Guidelines
- **Headings**: Clear hierarchy with proper nesting
- **Body text**: Scannable with good line height (1.5-1.6)
- **Labels**: Descriptive and concise
- **Placeholder text**: Helpful examples, not instructions
- **Error messages**: Specific, actionable, and friendly

---

*These UI guidelines ensure SemanticScout delivers a professional, accessible, and delightful user experience suitable for enterprise demonstrations and client presentations.*