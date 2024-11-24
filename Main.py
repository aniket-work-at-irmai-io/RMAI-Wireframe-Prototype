import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List
from streamlit_plotly_events import plotly_events

# Set page config
st.set_page_config(page_title="IRMAI System", layout="wide")


# Helper function for creating hierarchical knowledge graph
def create_hierarchical_knowledge_graph():
    # Create base graph
    G = nx.Graph()

    # Define node hierarchies
    hierarchies = {
        'Tier1_Risk1': ['Control1', 'Control2', 'Standard1'],
        'Tier1_Risk2': ['Control3', 'Control4', 'Standard2'],
        'Tier1_Risk3': ['Control5', 'Control6', 'Standard3']
    }

    # Add all nodes with their types
    node_types = {}
    for risk, children in hierarchies.items():
        G.add_node(risk, type='risk')
        node_types[risk] = 'risk'
        for child in children:
            G.add_node(child, type='control' if 'Control' in child else 'standard')
            node_types[child] = 'control' if 'Control' in child else 'standard'
            G.add_edge(risk, child)

    # Layout calculation
    pos = nx.spring_layout(G)

    # Create node traces by type
    colors = {'risk': 'red', 'control': 'blue', 'standard': 'green'}
    traces = []

    # Store node positions for click events
    node_positions = {}

    for node_type in ['risk', 'control', 'standard']:
        node_x = []
        node_y = []
        node_text = []
        node_ids = []

        for node in G.nodes():
            if node_types[node] == node_type:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_ids.append(node)
                node_positions[node] = (x, y)

        traces.append(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                marker=dict(size=30, color=colors[node_type]),
                text=node_text,
                textposition="top center",
                name=node_type.capitalize(),
                hoverinfo='text',
                customdata=node_ids
            )
        )

    # Add edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    traces.append(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
    )

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            title="Click on Tier 1 Risk nodes to expand/collapse"
        )
    )

    return fig, node_positions


# Enhanced Process Map function
def create_process_map(filters: Dict[str, str]):
    # Simulate process data based on filters
    processes = pd.DataFrame({
        'Step': range(1, 6),
        'Description': [f'Process Step {i}' for i in range(1, 6)],
        'Department': np.random.choice(['IT', 'Finance', 'Operations'], 5),
        'Geography': np.random.choice(['APAC', 'EMEA', 'Americas'], 5),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C'], 5),
        'Status': np.random.choice(['Compliant', 'Non-Compliant'], 5, p=[0.8, 0.2])
    })

    # Apply filters
    for key, value in filters.items():
        if value != "All":
            processes = processes[processes[key] == value]

    # Create Sankey diagram
    nodes = pd.DataFrame({
        'label': processes['Description'].tolist()
    })

    links = {
        'source': [],
        'target': [],
        'value': [],
        'color': []
    }

    for i in range(len(processes) - 1):
        links['source'].append(i)
        links['target'].append(i + 1)
        links['value'].append(1)
        links['color'].append('green' if processes.iloc[i]['Status'] == 'Compliant' else 'red')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes['label'],
            color="blue"
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color']
        )
    )])

    fig.update_layout(
        title_text=f"Process Map ({filters['Geography']} - {filters['Department']} - {filters['Product']})")
    return fig


# Digital Twin Module
def render_digital_twin():
    st.header("Digital Twin Module")

    # Initialize session state for expanded nodes if not exists
    if 'expanded_nodes' not in st.session_state:
        st.session_state.expanded_nodes = set()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Knowledge Graph Overview")
        # Create hierarchical knowledge graph
        fig, node_positions = create_hierarchical_knowledge_graph()

        # Display graph and handle click events
        selected_point = plotly_events(fig, click_event=True)
        if selected_point:
            node_id = selected_point[0]['customdata']
            if 'Tier1_Risk' in node_id:
                if node_id in st.session_state.expanded_nodes:
                    st.session_state.expanded_nodes.remove(node_id)
                else:
                    st.session_state.expanded_nodes.add(node_id)

        # Display expanded node details
        if st.session_state.expanded_nodes:
            with st.expander("Node Details", expanded=True):
                for node in st.session_state.expanded_nodes:
                    st.markdown(f"### {node}")

                    # Controls tab
                    st.markdown("#### Associated Controls")
                    controls_df = pd.DataFrame({
                        'Control ID': [f'{node}_CTR{i}' for i in range(1, 4)],
                        'Description': [f'Control for {node} - {i}' for i in range(1, 4)],
                        'Status': ['Active', 'Active', 'Under Review'],
                        'Last Review': ['2024-01-01', '2024-02-01', '2024-03-01']
                    })
                    st.dataframe(controls_df, hide_index=True)

                    # Standards tab
                    st.markdown("#### Associated Standards")
                    standards_df = pd.DataFrame({
                        'Standard ID': [f'{node}_STD{i}' for i in range(1, 4)],
                        'Description': [f'Standard for {node} - {i}' for i in range(1, 4)],
                        'Category': ['Regulatory', 'Industry', 'Internal'],
                        'Compliance': ['100%', '95%', '87%']
                    })
                    st.dataframe(standards_df, hide_index=True)

                    # Risk Metrics
                    metric_cols = st.columns(3)
                    metric_cols[0].metric(
                        "Risk Level",
                        "High",
                        delta="↑ Increased",
                        delta_color="inverse"
                    )
                    metric_cols[1].metric(
                        "Control Coverage",
                        "85%",
                        delta="3%"
                    )
                    metric_cols[2].metric(
                        "Standard Compliance",
                        "92%",
                        delta="-2%",
                        delta_color="inverse"
                    )

    with col2:
        st.subheader("Standards Repository")
        standards_tab1, standards_tab2, standards_tab3 = st.tabs([
            "Standards",
            "Validation Queue",
            "Recent Updates"
        ])

        with standards_tab1:
            # Category filter
            selected_category = st.selectbox(
                "Category",
                ["All Standards", "Regulatory Standards", "Industry Standards", "Internal Standards"]
            )

            # Search bar
            search_term = st.text_input("Search Standards", "")

            # Standards table
            standards_df = pd.DataFrame({
                'Standard ID': ['STD001', 'STD002', 'STD003', 'STD004', 'STD005'],
                'Category': ['Regulatory', 'Industry', 'Internal', 'Regulatory', 'Industry'],
                'Status': ['Active', 'Active', 'Under Review', 'Active', 'Deprecated'],
                'Last Updated': ['2024-03-01', '2024-02-15', '2024-03-10', '2024-01-20', '2024-02-28']
            })

            # Filter by category
            if selected_category != "All Standards":
                category = selected_category.split()[0]  # Get first word
                standards_df = standards_df[standards_df['Category'] == category]

            # Filter by search term
            if search_term:
                standards_df = standards_df[
                    standards_df.astype(str).apply(
                        lambda x: x.str.contains(search_term, case=False)
                    ).any(axis=1)
                ]

            st.dataframe(standards_df, hide_index=True, use_container_width=True)

        with standards_tab2:
            # Validation queue metrics
            st.metric("Pending Updates", "3", "2 urgent")
            st.metric("Recently Approved", "5", "Last 7 days")

            # Pending validations table
            validations_df = pd.DataFrame({
                'Standard ID': ['STD006', 'STD007', 'STD008'],
                'Type': ['New', 'Update', 'Deprecation'],
                'Priority': ['High', 'Medium', 'Low'],
                'Submitted': ['2024-03-09', '2024-03-08', '2024-03-07']
            })
            st.dataframe(validations_df, hide_index=True, use_container_width=True)

            # Action buttons for each validation
            for idx, row in validations_df.iterrows():
                cols = st.columns([2, 1, 1])
                cols[1].button("Approve", key=f"approve_{idx}")
                cols[2].button("Reject", key=f"reject_{idx}")

        with standards_tab3:
            # Recent updates timeline
            st.write("Recent Updates")
            updates = [
                {"date": "2024-03-10", "action": "Updated STD003 compliance requirements"},
                {"date": "2024-03-08", "action": "Added new control to STD002"},
                {"date": "2024-03-05", "action": "Deprecated STD005"},
                {"date": "2024-03-01", "action": "Created new standard STD008"}
            ]

            for update in updates:
                with st.expander(f"{update['date']} - {update['action'][:30]}..."):
                    st.write(update['action'])
                    st.write(f"Date: {update['date']}")

    # Bottom section for validation workflow
    st.subheader("Validation Workflow")
    workflow_cols = st.columns(4)

    workflow_cols[0].metric(
        "Total Standards",
        "156",
        delta="5 new this month"
    )
    workflow_cols[1].metric(
        "Active Standards",
        "142",
        delta="3 new"
    )
    workflow_cols[2].metric(
        "Under Review",
        "8",
        delta="-2"
    )
    workflow_cols[3].metric(
        "Deprecated",
        "6",
        delta="2"
    )


# Process Discovery Module
def render_process_discovery():
    st.header("Process Discovery Module")

    # Process Map Filters
    st.subheader("Process Map with Standards Overlay")

    filter_cols = st.columns(3)
    with filter_cols[0]:
        geography = st.selectbox("Geography", ["All", "APAC", "EMEA", "Americas"])
    with filter_cols[1]:
        department = st.selectbox("Department", ["All", "IT", "Finance", "Operations"])
    with filter_cols[2]:
        product = st.selectbox("Product", ["All", "Product A", "Product B", "Product C"])

    # Create and display process map based on filters
    filters = {
        "Geography": geography,
        "Department": department,
        "Product": product
    }

    process_map = create_process_map(filters)
    st.plotly_chart(process_map, use_container_width=True)

    # Process metrics based on filters
    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Total Steps", "5")
    metrics_cols[1].metric("Compliant Steps", "4")
    metrics_cols[2].metric("Non-Compliant Steps", "1")
    metrics_cols[3].metric("Compliance Rate", "80%")


# Risk Assessment Module
def render_risk_assessment():
    st.header("Risk Assessment Module")

    # Risk Overview
    st.subheader("Risk Overview Dashboard")
    risk_cols = st.columns(3)
    risk_cols[0].metric("High Risk", "3", delta="2")
    risk_cols[1].metric("Medium Risk", "8", delta="-1")
    risk_cols[2].metric("Low Risk", "12", delta="5")

    # Predictive Analytics
    st.subheader("Predictive Risk Analytics")
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Operational', 'Compliance', 'Financial']
    )
    st.line_chart(chart_data)

    # Risk Categories
    st.subheader("Risk Assessment Details")
    risk_tabs = st.tabs(["Operational", "Compliance", "Financial", "Strategic"])
    for tab in risk_tabs:
        with tab:
            st.dataframe(pd.DataFrame({
                'Risk Factor': [f'Factor {i}' for i in range(1, 4)],
                'Score': np.random.randint(1, 100, 3),
                'Trend': ['↑', '↓', '→']
            }))


# Controls Assessment Module
def render_controls_assessment():
    st.header("Controls Assessment Module")

    # Control Health Overview
    st.subheader("Control Health Dashboard")
    control_cols = st.columns(2)
    control_cols[0].metric("Passing Controls", "85%", delta="3%")
    control_cols[1].metric("Failed Controls", "15%", delta="-3%")

    # Framework Status
    st.subheader("Framework Compliance Status")
    frameworks = st.tabs(["SOC 2", "GDPR", "ISO 27001"])
    for framework in frameworks:
        with framework:
            st.progress(np.random.randint(70, 100) / 100)
            st.dataframe(pd.DataFrame({
                'Control': [f'Control {i}' for i in range(1, 4)],
                'Status': np.random.choice(['Pass', 'Fail', 'In Progress'], 3),
                'Last Tested': pd.date_range(start='2024-01-01', periods=3).strftime('%Y-%m-%d')
            }))

    # Testing Interface
    st.subheader("Automated Testing Interface")
    test_cols = st.columns(3)
    test_cols[0].metric("Active Tests", "12")
    test_cols[1].metric("Completed Tests", "45")
    test_cols[2].metric("Pending Validation", "8")


# Sidebar navigation
st.sidebar.title("IRMAI System")
selected_module = st.sidebar.radio(
    "Select Module",
    ["Digital Twin", "Process Discovery", "Risk Assessment", "Controls Assessment"]
)

# Render selected module
if selected_module == "Digital Twin":
    render_digital_twin()
elif selected_module == "Process Discovery":
    render_process_discovery()
elif selected_module == "Risk Assessment":
    render_risk_assessment()
else:
    render_controls_assessment()