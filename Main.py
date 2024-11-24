import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="IRMAI System", layout="wide")

# Sidebar navigation
st.sidebar.title("IRMAI System")
selected_module = st.sidebar.radio(
    "Select Module",
    ["Digital Twin", "Process Discovery", "Risk Assessment", "Controls Assessment"]
)


# Helper function for knowledge graph
def create_knowledge_graph():
    G = nx.random_geometric_graph(20, 0.3)
    pos = nx.spring_layout(G)
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color='#888'), mode='lines')
    node_trace = go.Scatter(
        x=[], y=[], mode='markers+text',
        marker=dict(size=20, color='lightblue'),
        text=[], textposition="top center")

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([f'Node {node}'])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig


# Digital Twin Module
def render_digital_twin():
    st.header("Digital Twin Module")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Knowledge Graph Overview")
        st.plotly_chart(create_knowledge_graph(), use_container_width=True)

    with col2:
        st.subheader("Standards Repository")
        standards_tab1, standards_tab2 = st.tabs(["Standards", "Validation Queue"])

        with standards_tab1:
            st.selectbox("Category", ["Regulatory Standards", "Industry Standards", "Internal Standards"])
            st.dataframe(pd.DataFrame({
                'Standard': ['STD001', 'STD002', 'STD003'],
                'Status': ['Active', 'Active', 'Under Review']
            }))

        with standards_tab2:
            st.metric("Pending Updates", "3")
            st.metric("Recently Approved", "5")


# Process Discovery Module
def render_process_discovery():
    st.header("Process Discovery Module")

    # Process Map
    st.subheader("Process Map with Standards Overlay")
    process_cols = st.columns(4)
    for i, status in enumerate(['Compliant', 'Non-Compliant', 'Under Review', 'Total']):
        process_cols[i].metric(
            status,
            np.random.randint(10, 100),
            delta=f"{np.random.randint(-10, 10)}%"
        )

    # Dummy process flow diagram
    st.area_chart(np.random.randn(20, 3))

    # Anomalies Dashboard
    st.subheader("Anomalies Dashboard")
    anomaly_cols = st.columns(3)
    anomaly_cols[0].error("Critical Anomalies: 2")
    anomaly_cols[1].warning("Warning Anomalies: 5")
    anomaly_cols[2].info("Under Review: 3")


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


# Render selected module
if selected_module == "Digital Twin":
    render_digital_twin()
elif selected_module == "Process Discovery":
    render_process_discovery()
elif selected_module == "Risk Assessment":
    render_risk_assessment()
else:
    render_controls_assessment()